"""Test ASK flow when user accepts: tool is forwarded, not asked again."""

import os
import sys
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest

from mcp_intent_proxy.cache import tool_signature
from mcp_intent_proxy.classifier import Classifier
from mcp_intent_proxy.logging import DecisionLog
from mcp_intent_proxy.proxy import build_server
from mcp_intent_proxy.rules import Decision, RuleTable
from mcp_intent_proxy.taxonomy import IntentResult

UPSTREAM = str(Path(__file__).parent / "upstream_stub.py")


async def test_accept_forwards_and_remembers(tmp_path: Path):
    """User accepts → tool executes; second call skips elicitation."""
    mock_client = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    clf = Classifier(client=mock_client)

    sig = tool_signature("echo", "Echo the input text back.", {
        "properties": {"text": {"title": "Text", "type": "string"}},
        "required": ["text"],
        "title": "echoArguments",
        "type": "object",
    })
    clf.cache.put(sig, IntentResult(
        action=["SEND"],
        sensitivity="personal-communications",
        externality="external",
        confidence=0.9,
        rationale="test",
    ))

    rules = RuleTable({"SEND": Decision.ASK})
    log = DecisionLog(log_dir=tmp_path)

    elicit_call_count = 0

    async def _accept(*args, **kwargs):
        nonlocal elicit_call_count
        elicit_call_count += 1
        return "allow_once"

    params = StdioServerParameters(
        command=sys.executable, args=[UPSTREAM], env=dict(os.environ)
    )
    async with stdio_client(params) as (up_read, up_write):
        async with ClientSession(up_read, up_write) as upstream:
            await upstream.initialize()
            server = build_server(
                upstream, log, classifier=clf, rule_table=rules,
                server_name="stub", elicit_fn=_accept,
            )

            list_req = ListToolsRequest(method="tools/list", params=None)
            await server.request_handlers[ListToolsRequest](list_req)

            # First call: user is asked and accepts.
            call_req = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name="echo", arguments={"text": "hi"}),
            )
            result = await server.request_handlers[CallToolRequest](call_req)
            assert result.root.isError is not True
            assert result.root.content[0].text == "hi"
            assert elicit_call_count == 1

            # Second call: should NOT elicit again (session memory).
            result2 = await server.request_handlers[CallToolRequest](call_req)
            assert result2.root.isError is not True
            assert result2.root.content[0].text == "hi"
            assert elicit_call_count == 1  # Still 1 — not asked again.

            # Rule table should NOT have been modified.
            assert rules.rules.get("SEND") == Decision.ASK

    log.close()
