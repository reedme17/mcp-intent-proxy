"""Test single-deny generalization: one ASK/deny writes a category-level rule
that then applies to other tools sharing the same intent category."""

import json
import sys
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ListToolsRequest,
)

from mcp_intent_proxy.cache import tool_signature
from mcp_intent_proxy.classifier import Classifier
from mcp_intent_proxy.logging import DecisionLog
from mcp_intent_proxy.proxy import build_server
from mcp_intent_proxy.rules import Decision, RuleTable
from mcp_intent_proxy.taxonomy import IntentResult

import os

UPSTREAM = str(Path(__file__).parent / "upstream_stub.py")


async def test_ask_generalizes_to_deny(tmp_path: Path):
    """
    Scenario: 'echo' is classified as SEND with rule=ask.
    On first call, the proxy generalizes to deny SEND globally.
    On second call to 'add' (also SEND), it is auto-denied without asking.
    """
    mock_client = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    clf = Classifier(client=mock_client)

    # Both tools classified as SEND.
    echo_sig = tool_signature("echo", "Echo the input text back.", {
        "properties": {"text": {"title": "Text", "type": "string"}},
        "required": ["text"],
        "title": "echoArguments",
        "type": "object",
    })
    add_sig = tool_signature("add", "Add two integers.", {
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "integer"},
        },
        "required": ["a", "b"],
        "title": "addArguments",
        "type": "object",
    })
    send_result = IntentResult(
        action=["SEND"],
        sensitivity="personal-communications",
        externality="external",
        confidence=0.9,
        rationale="test",
    )
    clf.cache.put(echo_sig, send_result)
    clf.cache.put(add_sig, send_result)

    # Initial rules: SEND is ask (not deny).
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text("SEND: ask\n")
    rules = RuleTable.load(rules_path)

    log = DecisionLog(log_dir=tmp_path)

    params = StdioServerParameters(
        command=sys.executable,
        args=[UPSTREAM],
        env=dict(os.environ),
    )

    async with stdio_client(params) as (up_read, up_write):
        async with ClientSession(up_read, up_write) as upstream:
            await upstream.initialize()

            # Point rule_table save at tmp_path.
            rules._RuleTable__save_path = rules_path  # noqa: not needed, save() takes path
            # Override DEFAULT_RULES_PATH for this test.
            import mcp_intent_proxy.rules as rules_mod
            original_default = rules_mod.DEFAULT_RULES_PATH
            rules_mod.DEFAULT_RULES_PATH = rules_path

            async def _always_decline(*args, **kwargs):
                return "decline"

            try:
                server = build_server(
                    upstream, log,
                    classifier=clf,
                    rule_table=rules,
                    server_name="stub",
                    elicit_fn=_always_decline,
                )

                # List tools to populate registry.
                list_req = ListToolsRequest(method="tools/list", params=None)
                await server.request_handlers[ListToolsRequest](list_req)

                # Call 'echo' — rule is ASK, so generalization fires.
                call_req = CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="echo", arguments={"text": "hi"}),
                )
                result = await server.request_handlers[CallToolRequest](call_req)
                assert result.root.isError is True
                assert "Blocked" in result.root.content[0].text

                # Verify the rule was generalized: SEND is now DENY in the table.
                assert rules.rules.get("SEND") == Decision.DENY

                # Now call 'add' — should be auto-denied by the new category rule.
                call_req2 = CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="add", arguments={"a": 1, "b": 2}),
                )
                result2 = await server.request_handlers[CallToolRequest](call_req2)
                assert result2.root.isError is True
                assert "Blocked" in result2.root.content[0].text

            finally:
                rules_mod.DEFAULT_RULES_PATH = original_default

    log.close()

    # Verify the rules file was persisted to disk.
    reloaded = RuleTable.load(rules_path)
    assert reloaded.rules.get("SEND") == Decision.DENY

    # Verify log shows ask (generalized) then deny.
    lines = [json.loads(l) for l in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert lines[0]["decision"] == "ask"
    assert lines[1]["decision"] == "deny"
