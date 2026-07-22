"""End-to-end enforcement: proxy denies tools/call based on classified intent + rules."""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from mcp_intent_proxy.classifier import Classifier
from mcp_intent_proxy.proxy import build_server
from mcp_intent_proxy.rules import Decision, RuleTable
from mcp_intent_proxy.taxonomy import IntentResult
from mcp_intent_proxy.logging import DecisionLog

import anyio
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server


UPSTREAM = str(Path(__file__).parent / "upstream_stub.py")


async def test_deny_returns_error(tmp_path: Path):
    """When a tool's intent matches a deny rule, proxy returns isError."""
    # Set up a classifier that always returns DELETE intent.
    mock_client = MagicMock()
    clf = Classifier(client=mock_client)
    delete_result = IntentResult(
        action=["DELETE"],
        sensitivity="non-specific",
        externality="internal",
        confidence=0.95,
        rationale="test",
    )
    # Pre-populate cache so classify() won't call the LLM.
    from mcp_intent_proxy.cache import tool_signature
    sig = tool_signature("echo", "Echo the input text back.", {
        "properties": {"text": {"title": "Text", "type": "string"}},
        "required": ["text"],
        "title": "echoArguments",
        "type": "object",
    })
    clf.cache.put(sig, delete_result)

    # Rule table: deny DELETE.
    rules = RuleTable({"DELETE": Decision.DENY})
    log = DecisionLog(log_dir=tmp_path)

    # Connect to upstream stub and build the proxy server.
    import os
    params = StdioServerParameters(
        command=sys.executable,
        args=[UPSTREAM],
        env=dict(os.environ),
    )

    async with stdio_client(params) as (up_read, up_write):
        async with ClientSession(up_read, up_write) as upstream:
            await upstream.initialize()
            server = build_server(
                upstream, log,
                classifier=clf,
                rule_table=rules,
                server_name="stub",
            )

            # Simulate a client calling tools/list then tools/call via the server.
            # We'll use the server's handlers directly (unit-level).
            from mcp.types import ListToolsRequest, CallToolRequest, CallToolRequestParams

            # First list tools to populate registry.
            list_req = ListToolsRequest(method="tools/list", params=None)
            await server.request_handlers[ListToolsRequest](list_req)

            # Now call 'echo' — should be denied.
            call_req = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name="echo", arguments={"text": "hi"}),
            )
            result = await server.request_handlers[CallToolRequest](call_req)
            call_result = result.root
            assert call_result.isError is True
            assert "Blocked by user policy" in call_result.content[0].text
            assert "DELETE" in call_result.content[0].text

    # Verify log recorded the deny decision.
    log.close()
    lines = [json.loads(l) for l in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert any(l["decision"] == "deny" for l in lines)


async def test_allow_forwards_normally(tmp_path: Path):
    """When a tool's intent matches an allow rule, call is forwarded."""
    mock_client = MagicMock()
    clf = Classifier(client=mock_client)
    read_result = IntentResult(
        action=["READ/SEARCH"],
        sensitivity="non-specific",
        externality="internal",
        confidence=0.95,
        rationale="test",
    )
    from mcp_intent_proxy.cache import tool_signature
    sig = tool_signature("echo", "Echo the input text back.", {
        "properties": {"text": {"title": "Text", "type": "string"}},
        "required": ["text"],
        "title": "echoArguments",
        "type": "object",
    })
    clf.cache.put(sig, read_result)

    # Explicit allow: under fail-closed defaults an unregulated category asks.
    rules = RuleTable({"DELETE": Decision.DENY, "READ/SEARCH": Decision.ALLOW})
    log = DecisionLog(log_dir=tmp_path)

    import os
    params = StdioServerParameters(
        command=sys.executable,
        args=[UPSTREAM],
        env=dict(os.environ),
    )

    async with stdio_client(params) as (up_read, up_write):
        async with ClientSession(up_read, up_write) as upstream:
            await upstream.initialize()
            server = build_server(
                upstream, log,
                classifier=clf,
                rule_table=rules,
                server_name="stub",
            )

            from mcp.types import ListToolsRequest, CallToolRequest, CallToolRequestParams

            list_req = ListToolsRequest(method="tools/list", params=None)
            await server.request_handlers[ListToolsRequest](list_req)

            call_req = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name="echo", arguments={"text": "hi"}),
            )
            result = await server.request_handlers[CallToolRequest](call_req)
            call_result = result.root
            assert call_result.isError is not True
            assert call_result.content[0].text == "hi"

    log.close()
    lines = [json.loads(l) for l in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert all(l["decision"] == "allow" for l in lines)
