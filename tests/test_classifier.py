"""Unit tests for the classifier: prompt assembly, parsing, caching, fail-closed."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mcp_intent_proxy.classifier import Classifier, _extract_json_block
from mcp_intent_proxy.taxonomy import FAIL_CLOSED_RESULT, IntentResult


VALID_LLM_RESPONSE = """\
This tool permanently removes a file from the filesystem. The input schema \
requires a file path, and the operation is described as irreversible deletion.

```json
{
  "action": ["DELETE"],
  "sensitivity": "non-specific",
  "externality": "internal",
  "confidence": 0.95,
  "rationale": "Deletes a file at a given path; irreversible destructive operation."
}
```
"""

MULTI_ACTION_RESPONSE = """\
This tool sends a payment notification email to a customer after processing \
a charge. It both spends money and sends outbound communications.

```json
{
  "action": ["SPEND", "SEND"],
  "sensitivity": "financial",
  "externality": "external",
  "confidence": 0.88,
  "rationale": "Charges a payment method and emails receipt to external recipient."
}
```
"""


def _mock_client(response_text: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=response_text)]
    client.messages.create.return_value = msg
    return client


class TestParsing:
    def test_extract_json_block(self):
        raw = _extract_json_block(VALID_LLM_RESPONSE)
        import json
        data = json.loads(raw)
        assert data["action"] == ["DELETE"]
        assert data["confidence"] == 0.95

    def test_extract_json_no_fence(self):
        text = 'Some preamble {"action": ["READ/SEARCH"], "sensitivity": "non-specific", "externality": "internal", "confidence": 0.9, "rationale": "reads"} trailing'
        raw = _extract_json_block(text)
        import json
        data = json.loads(raw)
        assert data["action"] == ["READ/SEARCH"]

    def test_extract_json_raises_on_empty(self):
        with pytest.raises(ValueError):
            _extract_json_block("no json here at all")


class TestClassifier:
    def test_basic_classification(self):
        client = _mock_client(VALID_LLM_RESPONSE)
        clf = Classifier(client=client)
        result = clf.classify(
            tool_name="delete_file",
            tool_description="Permanently delete a file.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        assert result.action == ["DELETE"]
        assert result.sensitivity == "non-specific"
        assert result.externality == "internal"
        assert result.confidence == 0.95
        assert "irreversible" in result.rationale

    def test_multi_action(self):
        client = _mock_client(MULTI_ACTION_RESPONSE)
        clf = Classifier(client=client)
        result = clf.classify(
            tool_name="charge_and_notify",
            tool_description="Process payment and email receipt.",
            input_schema={"type": "object", "properties": {"amount": {"type": "number"}}},
        )
        assert set(result.action) == {"SPEND", "SEND"}
        assert result.sensitivity == "financial"
        assert result.externality == "external"

    def test_cache_hit(self):
        client = _mock_client(VALID_LLM_RESPONSE)
        clf = Classifier(client=client)
        clf.classify(
            tool_name="rm",
            tool_description="Remove file.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        clf.classify(
            tool_name="rm",
            tool_description="Remove file.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        assert client.messages.create.call_count == 1

    def test_fail_closed_on_parse_error(self):
        client = _mock_client("I don't know how to classify this tool.")
        clf = Classifier(client=client)
        result = clf.classify(
            tool_name="mystery",
            tool_description="???",
            input_schema={},
        )
        assert result.confidence == 0.0
        assert result.action == FAIL_CLOSED_RESULT.action

    def test_fail_closed_on_low_confidence(self):
        low_conf = """\
Unclear tool.

```json
{
  "action": ["READ/SEARCH"],
  "sensitivity": "non-specific",
  "externality": "internal",
  "confidence": 0.3,
  "rationale": "Not sure"
}
```
"""
        client = _mock_client(low_conf)
        clf = Classifier(client=client, confidence_threshold=0.6)
        result = clf.classify(
            tool_name="ambiguous",
            tool_description="Does something.",
            input_schema={},
        )
        assert result.confidence == 0.0  # fail-closed replaces it

    def test_fail_closed_on_invalid_taxonomy(self):
        bad_taxonomy = """\
Some reasoning.

```json
{
  "action": ["INVALID_ACTION"],
  "sensitivity": "made-up",
  "externality": "internal",
  "confidence": 0.99,
  "rationale": "bad"
}
```
"""
        client = _mock_client(bad_taxonomy)
        clf = Classifier(client=client)
        result = clf.classify(
            tool_name="x",
            tool_description="x",
            input_schema={},
        )
        assert result.confidence == 0.0

    def test_server_context_included_in_prompt(self):
        client = _mock_client(VALID_LLM_RESPONSE)
        clf = Classifier(client=client, include_server_context=True)
        clf.classify(
            tool_name="delete_file",
            tool_description="Delete.",
            input_schema={},
            server_name="filesystem",
            server_description="Local filesystem access",
        )
        call_args = client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "Server: filesystem" in user_msg
        assert "Local filesystem access" in user_msg

    def test_server_context_excluded_when_disabled(self):
        client = _mock_client(VALID_LLM_RESPONSE)
        clf = Classifier(client=client, include_server_context=False)
        clf.classify(
            tool_name="delete_file",
            tool_description="Delete.",
            input_schema={},
            server_name="filesystem",
            server_description="Local filesystem access",
        )
        call_args = client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "Server:" not in user_msg

    def test_llm_exception_returns_fail_closed(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        clf = Classifier(client=client)
        result = clf.classify(
            tool_name="x",
            tool_description="x",
            input_schema={},
        )
        assert result.confidence == 0.0


class TestClassifyBatch:
    def test_batch_classifies_all(self):
        client = _mock_client(VALID_LLM_RESPONSE)
        clf = Classifier(client=client)
        tools = [
            {"name": "a", "description": "Delete a.", "inputSchema": {}},
            {"name": "b", "description": "Delete b.", "inputSchema": {"x": 1}},
        ]
        results = clf.classify_batch(tools, server_name="test")
        assert len(results) == 2
        assert client.messages.create.call_count == 2

    def test_batch_uses_cache(self):
        client = _mock_client(VALID_LLM_RESPONSE)
        clf = Classifier(client=client)
        tools = [
            {"name": "a", "description": "Delete.", "inputSchema": {}},
            {"name": "a", "description": "Delete.", "inputSchema": {}},
        ]
        results = clf.classify_batch(tools)
        assert len(results) == 2
        assert client.messages.create.call_count == 1
