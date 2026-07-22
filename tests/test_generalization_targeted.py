"""Verify generalization only targets trigger labels, not all labels on a tool.

Scenario: a tool classified as [READ/SEARCH, SEND] with rule SEND=ask.
After generalization, SEND should become deny but READ/SEARCH must stay allow.
"""

import json
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


async def test_only_trigger_label_generalized(tmp_path: Path):
    """SEND=ask should generalize deny for SEND only, not READ/SEARCH."""
    import mcp_intent_proxy.rules as rules_mod

    mock_client = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    clf = Classifier(client=mock_client)

    # Tool has BOTH READ/SEARCH and SEND.
    multi_intent = IntentResult(
        action=["READ/SEARCH", "SEND"],
        sensitivity="personal-communications",
        externality="external",
        confidence=0.9,
        rationale="reads contacts and sends notification",
    )
    sig = tool_signature("echo", "Echo the input text back.", {
        "properties": {"text": {"title": "Text", "type": "string"}},
        "required": ["text"],
        "title": "echoArguments",
        "type": "object",
    })
    clf.cache.put(sig, multi_intent)

    # Rule: only SEND is ask. READ/SEARCH is unregulated (defaults to ask too).
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text("SEND: ask\n")
    rules = RuleTable.load(rules_path)

    original_default = rules_mod.DEFAULT_RULES_PATH
    rules_mod.DEFAULT_RULES_PATH = rules_path

    log = DecisionLog(log_dir=tmp_path)

    try:
        params = StdioServerParameters(
            command=sys.executable, args=[UPSTREAM], env=dict(os.environ)
        )

        # Per-category answers: user tolerates READ/SEARCH this once but
        # objects to SEND. Only SEND may be generalized.
        async def _per_label(tool_name, trigger_labels, intent):
            label = trigger_labels[0]
            if label == "SEND":
                return "never_category"
            return "allow_once"

        async with stdio_client(params) as (up_read, up_write):
            async with ClientSession(up_read, up_write) as upstream:
                await upstream.initialize()
                server = build_server(
                    upstream, log, classifier=clf, rule_table=rules,
                    server_name="stub", elicit_fn=_per_label,
                )

                # Populate registry.
                list_req = ListToolsRequest(method="tools/list", params=None)
                await server.request_handlers[ListToolsRequest](list_req)

                # Call echo — asks per category; user denies SEND.
                call_req = CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="echo", arguments={"text": "hi"}),
                )
                result = await server.request_handlers[CallToolRequest](call_req)
                assert result.root.isError is True

                # Verify ONLY SEND was generalized to deny.
                assert rules.rules.get("SEND") == Decision.DENY
                assert "READ/SEARCH" not in rules.rules  # Must NOT have been touched.

    finally:
        rules_mod.DEFAULT_RULES_PATH = original_default
        log.close()

    # Verify persisted file only has SEND: deny.
    reloaded = RuleTable.load(rules_path)
    assert reloaded.rules.get("SEND") == Decision.DENY
    assert "READ/SEARCH" not in reloaded.rules
