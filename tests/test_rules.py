"""Tests for rule table loading, evaluation, and persistence."""

from pathlib import Path

import pytest

from mcp_intent_proxy.rules import Decision, RuleTable


class TestEvaluation:
    def test_empty_table_allows_all(self):
        rt = RuleTable()
        assert rt.evaluate(["DELETE"], "non-specific", "internal") == Decision.ALLOW

    def test_single_action_deny(self):
        rt = RuleTable({"DELETE": Decision.DENY})
        assert rt.evaluate(["DELETE"], "non-specific", "internal") == Decision.DENY

    def test_most_restrictive_wins(self):
        rt = RuleTable({"READ/SEARCH": Decision.ALLOW, "DELETE": Decision.DENY})
        assert rt.evaluate(["READ/SEARCH", "DELETE"], "non-specific", "internal") == Decision.DENY

    def test_ask_beats_allow(self):
        rt = RuleTable({"SEND": Decision.ASK})
        assert rt.evaluate(["SEND"], "non-specific", "external") == Decision.ASK

    def test_deny_beats_ask(self):
        rt = RuleTable({"SEND": Decision.ASK, "SPEND": Decision.DENY})
        assert rt.evaluate(["SEND", "SPEND"], "financial", "external") == Decision.DENY

    def test_sensitivity_as_rule_key(self):
        rt = RuleTable({"FINANCIAL": Decision.DENY})
        assert rt.evaluate(["CREATE"], "financial", "internal") == Decision.DENY

    def test_externality_as_rule_key(self):
        rt = RuleTable({"EXTERNAL": Decision.ASK})
        assert rt.evaluate(["READ/SEARCH"], "non-specific", "external") == Decision.ASK

    def test_case_insensitive_lookup(self):
        rt = RuleTable({"delete": Decision.DENY})
        assert rt.evaluate(["DELETE"], "non-specific", "internal") == Decision.DENY


class TestYAML:
    def test_load_from_file(self, tmp_path: Path):
        rules_file = tmp_path / "rules.yaml"
        rules_file.write_text("DELETE: deny\nSEND: ask\nREAD/SEARCH: allow\n")
        rt = RuleTable.load(rules_file)
        assert rt.rules == {
            "DELETE": Decision.DENY,
            "SEND": Decision.ASK,
            "READ/SEARCH": Decision.ALLOW,
        }

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        rt = RuleTable.load(tmp_path / "nonexistent.yaml")
        assert rt.rules == {}

    def test_save_and_reload(self, tmp_path: Path):
        rt = RuleTable({"DELETE": Decision.DENY, "SPEND": Decision.DENY})
        path = tmp_path / "rules.yaml"
        rt.save(path)
        reloaded = RuleTable.load(path)
        assert reloaded.rules == rt.rules

    def test_invalid_decision_ignored(self, tmp_path: Path):
        rules_file = tmp_path / "rules.yaml"
        rules_file.write_text("DELETE: deny\nSEND: invalid_value\n")
        rt = RuleTable.load(rules_file)
        assert "DELETE" in rt.rules
        assert "SEND" not in rt.rules


class TestSetRule:
    def test_set_rule_adds_new(self):
        rt = RuleTable()
        rt.set_rule("SPEND", Decision.DENY)
        assert rt.evaluate(["SPEND"], "non-specific", "internal") == Decision.DENY

    def test_set_rule_overwrites(self):
        rt = RuleTable({"SPEND": Decision.ALLOW})
        rt.set_rule("SPEND", Decision.DENY)
        assert rt.evaluate(["SPEND"], "non-specific", "internal") == Decision.DENY
