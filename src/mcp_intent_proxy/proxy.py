"""Transparent stdio proxy between an MCP client and one upstream MCP server.

The proxy speaks MCP on stdin/stdout (server side) and spawns the upstream
server as a subprocess (client side). tools/list and tools/call are forwarded
verbatim; raw request handlers are registered instead of the SDK's decorator
wrappers so upstream results — including isError and structuredContent — pass
through unmodified.

Every tools/call interception is classified by intent and recorded in a JSONL
decision log. Classification happens eagerly at tools/list time (batch
pre-classify) so that tools/call decisions incur zero LLM latency.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.server import request_ctx
from mcp.server.stdio import stdio_server

from .classifier import Classifier
from .logging import DecisionLog
from .rules import Decision, RuleTable
from .taxonomy import IntentResult

logger = logging.getLogger("mcp_intent_proxy")

SERVER_NAME = "mcp-intent-proxy"


DENY_MESSAGE_TEMPLATE = (
    "Blocked by user policy. This tool was classified as intent={actions} "
    "(sensitivity={sensitivity}, externality={externality}). "
    "The user's rules deny this category of operations. "
    "Do not retry this call and do not attempt to accomplish the same goal "
    "through other tools. Inform the user that the action was blocked by "
    "their policy, and that they can change this by editing rules.yaml."
)


# Elicitation answers: what the user chose for an ASK decision.
ANSWER_ALLOW_ONCE = "allow_once"
ANSWER_ALWAYS_CATEGORY = "always_category"
ANSWER_NEVER_CATEGORY = "never_category"
ANSWER_CANCEL = "cancel"


async def _default_elicit(
    tool_name: str, trigger_labels: list[str], intent: IntentResult | None
) -> str:
    """Ask the user via MCP elicitation.

    Three-way choice so the rule table crystallizes per category during use:
    - allow_once: forward this call only, write no rule
    - always_category: write allow rules for the trigger categories
    - never_category: write deny rules for the trigger categories

    Returns one of the ANSWER_* constants.
    """
    categories = ", ".join(trigger_labels) if trigger_labels else "unknown"
    rationale = intent.rationale if intent else ""
    message = (
        f"The agent wants to use tool '{tool_name}' which is classified as "
        f"{categories}. {rationale}\n\n"
        f"Choose: allow this call only, always allow {categories} operations, "
        f"or never allow {categories} operations."
    )
    schema = {
        "type": "object",
        "properties": {
            "choice": {
                "type": "string",
                "title": f"Tool '{tool_name}' wants to run ({categories})",
                "oneOf": [
                    {"const": ANSWER_ALLOW_ONCE, "title": "Allow this call only"},
                    {"const": ANSWER_ALWAYS_CATEGORY, "title": f"Always allow {categories}"},
                    {"const": ANSWER_NEVER_CATEGORY, "title": f"Never allow {categories}"},
                ],
            }
        },
        "required": ["choice"],
    }
    try:
        ctx = request_ctx.get()
        result = await ctx.session.elicit(message=message, requestedSchema=schema)
        if result.action == "accept" and result.content:
            choice = result.content.get("choice")
            if choice in (ANSWER_ALLOW_ONCE, ANSWER_ALWAYS_CATEGORY, ANSWER_NEVER_CATEGORY):
                return choice
            return ANSWER_CANCEL
        elif result.action == "decline":
            return ANSWER_NEVER_CATEGORY
        else:
            return ANSWER_CANCEL
    except Exception as e:
        logger.warning("Elicitation failed (client may not support it): %s", e)
        return ANSWER_CANCEL


# Type alias for the elicitation callback (injectable for testing).
ElicitFn = Any  # async (tool_name, trigger_labels, intent) -> str (ANSWER_*)


def build_server(
    upstream: ClientSession,
    log: DecisionLog,
    classifier: Classifier | None = None,
    rule_table: RuleTable | None = None,
    server_name: str = "",
    server_description: str = "",
    elicit_fn: ElicitFn | None = None,
) -> Server:
    """Build a Server whose tool endpoints forward to the upstream session."""
    server = Server(SERVER_NAME)

    # tools/list: forward upstream result, and pre-classify all tools eagerly.
    async def _list_tools(request: types.ListToolsRequest) -> types.ServerResult:
        cursor = request.params.cursor if request.params else None
        result = await upstream.list_tools(cursor=cursor)
        if classifier is not None:
            _preclassify(classifier, result.tools, server_name, server_description)
        return types.ServerResult(result)

    # tools/call: classify (cache hit after pre-classify), check policy, enforce.
    async def _call_tool(request: types.CallToolRequest) -> types.ServerResult:
        name = request.params.name
        arguments = request.params.arguments or {}

        intent_dict: dict[str, Any] | None = None
        intent: IntentResult | None = None
        decision = Decision.ALLOW
        trigger_labels: list[str] = []
        late_registered = False
        mixed_envelope = False

        if classifier is not None:
            tool_meta = _find_tool_meta(name)
            if tool_meta is None:
                # Tool not seen via tools/list. Re-sync from upstream in case
                # the server legitimately added it after initialization.
                refreshed = await upstream.list_tools()
                _preclassify(classifier, refreshed.tools, server_name, server_description)
                tool_meta = _find_tool_meta(name)

            if tool_meta is None:
                # Still unknown after re-sync — upstream doesn't have it.
                log.record(
                    tool=name, arguments=arguments, decision="deny",
                    intent=None, rule="fail-closed",
                    flags=["unknown-tool"],
                )
                return types.ServerResult(
                    types.CallToolResult(
                        content=[types.TextContent(
                            type="text",
                            text="Blocked: this tool does not exist on the upstream "
                                 "server. Do not retry. Inform the user.",
                        )],
                        isError=True,
                    )
                )
            else:
                late_registered = name not in _ask_allowed and not _was_in_initial_list(name)

            intent = classifier.classify(
                tool_name=name,
                tool_description=tool_meta.get("description", ""),
                input_schema=tool_meta.get("inputSchema", {}),
                server_name=server_name,
                server_description=server_description,
            )
            intent_dict = intent.to_dict()
            if rule_table is not None:
                eval_result = rule_table.evaluate_detailed(
                    intent.action, intent.sensitivity, intent.externality
                )
                decision = eval_result.decision
                trigger_labels = eval_result.triggers
                mixed_envelope = eval_result.mixed
            # Late-registered tool: escalate to ASK unless already DENY.
            if late_registered and decision == Decision.ALLOW:
                decision = Decision.ASK
                trigger_labels = intent.action
                intent = IntentResult(
                    action=intent.action,
                    sensitivity=intent.sensitivity,
                    externality=intent.externality,
                    confidence=intent.confidence,
                    rationale=(
                        "[WARNING: this tool was not present when the server "
                        "started — it appeared after initialization.] "
                        + intent.rationale
                    ),
                )
                intent_dict = intent.to_dict()
                logger.warning(
                    "Late-registered tool '%s' not in initial tools/list; "
                    "escalating to ASK for user confirmation",
                    name,
                )

        decision_label = "forward" if classifier is None else decision.value
        flags = []
        if late_registered:
            flags.append("late-registered")
        log.record(
            tool=name,
            arguments=arguments,
            decision=decision_label,
            intent=intent_dict,
            rule=decision.value if classifier is not None and decision != Decision.ALLOW else None,
            flags=flags or None,
        )

        if decision == Decision.DENY:
            message = DENY_MESSAGE_TEMPLATE.format(
                actions=",".join(intent.action) if intent else "UNKNOWN",
                sensitivity=intent.sensitivity if intent else "unknown",
                externality=intent.externality if intent else "unknown",
            )
            return types.ServerResult(
                types.CallToolResult(
                    content=[types.TextContent(type="text", text=message)],
                    isError=True,
                )
            )

        if decision == Decision.ASK:
            # Skip elicitation if user already said yes this session.
            if name in _ask_allowed:
                result = await upstream.call_tool(name, arguments)
                return types.ServerResult(result)

            ask_labels = trigger_labels or (list(intent.action) if intent else [])

            # Mixed envelope: tell the user why this became a question.
            ask_intent = intent
            if mixed_envelope and intent is not None:
                ask_intent = IntentResult(
                    action=intent.action,
                    sensitivity=intent.sensitivity,
                    externality=intent.externality,
                    confidence=intent.confidence,
                    rationale=(
                        f"[This tool combines denied capabilities "
                        f"({', '.join(ask_labels)}) with permitted ones — "
                        f"a blanket block would break its benign uses.] "
                        + intent.rationale
                    ),
                )

            # One question per category: each answer settles exactly one
            # label, so every written rule is attributable to an explicit
            # user statement about that category (no bundled consent).
            # Interruptions are bounded by the number of categories — each
            # settled category never asks again.
            answers: dict[str, str] = {}
            for label in ask_labels:
                answer = await _elicit(name, [label], ask_intent)
                answers[label] = answer
                if answer == ANSWER_NEVER_CATEGORY:
                    # Call is blocked regardless; leave remaining categories
                    # unsettled rather than asking about a dead call.
                    break

            if rule_table is not None:
                wrote = False
                for label, answer in answers.items():
                    if answer == ANSWER_ALWAYS_CATEGORY:
                        rule_table.set_rule(label, Decision.ALLOW)
                        wrote = True
                    elif answer == ANSWER_NEVER_CATEGORY:
                        rule_table.set_rule(label, Decision.DENY)
                        wrote = True
                if wrote:
                    rule_table.save()
                    logger.info(
                        "Rules updated from user answers: tool=%s answers=%s",
                        name,
                        answers,
                    )

            denied_labels = [
                l for l, a in answers.items() if a == ANSWER_NEVER_CATEGORY
            ]
            cancelled_labels = [
                l for l, a in answers.items() if a == ANSWER_CANCEL
            ]

            if denied_labels:
                message = DENY_MESSAGE_TEMPLATE.format(
                    actions=",".join(denied_labels),
                    sensitivity=intent.sensitivity if intent else "unknown",
                    externality=intent.externality if intent else "unknown",
                )
                if mixed_envelope:
                    message += (
                        " Note: this tool was blocked because of its "
                        f"{','.join(denied_labels)} capability. If the task only "
                        "needs its other capabilities, use a narrower tool that "
                        "does not include the denied capability."
                    )
                return types.ServerResult(
                    types.CallToolResult(
                        content=[types.TextContent(type="text", text=message)],
                        isError=True,
                    )
                )

            if cancelled_labels:
                # No answer is not an opinion: deny this one call, write no
                # rule, ask again next time.
                return types.ServerResult(
                    types.CallToolResult(
                        content=[types.TextContent(
                            type="text",
                            text=f"Blocked: awaiting user confirmation for intent={','.join(cancelled_labels)}. "
                                 "The user did not respond. Do not retry.",
                        )],
                        isError=True,
                    )
                )

            # All labels answered allow-once or always-category: forward.
            # Remember allow-once answers for the session so the same tool
            # doesn't re-ask (always-category labels are settled by rule).
            if any(a == ANSWER_ALLOW_ONCE for a in answers.values()):
                _ask_allowed.add(name)
            result = await upstream.call_tool(name, arguments)
            return types.ServerResult(result)

        result = await upstream.call_tool(name, arguments)
        return types.ServerResult(result)

    # Tools the user has already said "yes" to during this session.
    _ask_allowed: set[str] = set()
    _elicit = elicit_fn or _default_elicit
    # Tools that were present in the first tools/list response.
    _initial_tools: set[str] = set()
    _initial_list_done = False

    def _was_in_initial_list(tool_name: str) -> bool:
        return tool_name in _initial_tools

    # Registry of known tools populated by _preclassify for lookup at call time.
    _tool_registry: dict[str, dict[str, Any]] = {}

    def _preclassify(
        clf: Classifier,
        tools: list[types.Tool],
        srv_name: str,
        srv_desc: str,
    ) -> None:
        nonlocal _initial_list_done
        batch = []
        for t in tools:
            meta = {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema or {},
            }
            _tool_registry[t.name] = meta
            batch.append(meta)
            if not _initial_list_done:
                _initial_tools.add(t.name)
        _initial_list_done = True
        clf.classify_batch(batch, server_name=srv_name, server_description=srv_desc)
        logger.info("Pre-classified %d tools from %s", len(batch), srv_name or "(unknown)")

    def _find_tool_meta(name: str) -> dict[str, Any] | None:
        return _tool_registry.get(name)

    server.request_handlers[types.ListToolsRequest] = _list_tools
    server.request_handlers[types.CallToolRequest] = _call_tool
    return server


async def run_proxy(
    command: str,
    args: list[str],
    *,
    enable_classifier: bool = True,
    include_server_context: bool = True,
    rules_path: str | None = None,
) -> None:
    """Spawn the upstream server and serve the proxy over stdio until EOF."""
    from pathlib import Path

    params = StdioServerParameters(command=command, args=args, env=dict(os.environ))
    log = DecisionLog()

    classifier: Classifier | None = None
    rule_table: RuleTable | None = None

    if enable_classifier:
        classifier = Classifier(include_server_context=include_server_context)
        rule_table = RuleTable.load(Path(rules_path) if rules_path else None)

    try:
        async with stdio_client(params) as (upstream_read, upstream_write):
            async with ClientSession(upstream_read, upstream_write) as upstream:
                init_result = await upstream.initialize()

                # Auto-detect server identity from the MCP handshake.
                server_name = ""
                server_description = ""
                if init_result.serverInfo:
                    server_name = init_result.serverInfo.name or ""
                server_description = init_result.instructions or ""
                logger.info(
                    "Connected to upstream: name=%r description=%r",
                    server_name,
                    server_description[:80],
                )

                server = build_server(
                    upstream,
                    log,
                    classifier=classifier,
                    rule_table=rule_table,
                    server_name=server_name,
                    server_description=server_description,
                )
                async with stdio_server() as (read_stream, write_stream):
                    await server.run(
                        read_stream,
                        write_stream,
                        server.create_initialization_options(),
                    )
    finally:
        log.close()
