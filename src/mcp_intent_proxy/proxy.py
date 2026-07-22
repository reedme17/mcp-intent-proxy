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
from mcp.server.stdio import stdio_server

from .classifier import Classifier
from .logging import DecisionLog
from .rules import Decision, RuleTable

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


def build_server(
    upstream: ClientSession,
    log: DecisionLog,
    classifier: Classifier | None = None,
    rule_table: RuleTable | None = None,
    server_name: str = "",
    server_description: str = "",
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
        decision = Decision.ALLOW

        if classifier is not None:
            tool_meta = _find_tool_meta(name)
            if tool_meta:
                intent = classifier.classify(
                    tool_name=name,
                    tool_description=tool_meta.get("description", ""),
                    input_schema=tool_meta.get("inputSchema", {}),
                    server_name=server_name,
                    server_description=server_description,
                )
                intent_dict = intent.to_dict()
                if rule_table is not None:
                    decision = rule_table.evaluate(
                        intent.action, intent.sensitivity, intent.externality
                    )

        decision_label = "forward" if classifier is None else decision.value
        log.record(
            tool=name,
            arguments=arguments,
            decision=decision_label,
            intent=intent_dict,
            rule=decision.value if classifier is not None and decision != Decision.ALLOW else None,
        )

        if decision == Decision.DENY:
            actions_str = ",".join(intent.action) if intent_dict else "UNKNOWN"
            sensitivity_str = intent.sensitivity if intent_dict else "unknown"
            externality_str = intent.externality if intent_dict else "unknown"
            message = DENY_MESSAGE_TEMPLATE.format(
                actions=actions_str,
                sensitivity=sensitivity_str,
                externality=externality_str,
            )
            return types.ServerResult(
                types.CallToolResult(
                    content=[types.TextContent(type="text", text=message)],
                    isError=True,
                )
            )

        # ASK is treated as allow for now (MVP; interactive prompt is future work).
        result = await upstream.call_tool(name, arguments)
        return types.ServerResult(result)

    # Registry of known tools populated by _preclassify for lookup at call time.
    _tool_registry: dict[str, dict[str, Any]] = {}

    def _preclassify(
        clf: Classifier,
        tools: list[types.Tool],
        srv_name: str,
        srv_desc: str,
    ) -> None:
        batch = []
        for t in tools:
            meta = {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema or {},
            }
            _tool_registry[t.name] = meta
            batch.append(meta)
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
    server_name: str = "",
    server_description: str = "",
    rules_path: str | None = None,
) -> None:
    """Spawn the upstream server and serve the proxy over stdio until EOF."""
    # Inherit the proxy's full environment: the SDK default is a minimal env,
    # which would strip variables the upstream server needs (PATH tweaks,
    # registry overrides, API keys).
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
                await upstream.initialize()
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
