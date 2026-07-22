"""Tests for rule table loading, evaluation, and persistence."""

from pathlib import Path

import pytest

from mcp_intent_proxy.rules import Decision, RuleTable


class TestEvaluation:
    def test_unregulated_action_asks(self):
        # Fail-closed default: no rule for the category → ask, not allow.
        rt = RuleTable()
        res = rt.evaluate_detailed(["DELETE"], "non-specific", "internal")
        assert res.decision == Decision.ASK
        assert res.triggers == ["DELETE"]
        assert res.mixed is False

    def test_explicit_allow(self):
        rt = RuleTable({"READ/SEARCH": Decision.ALLOW})
        assert rt.evaluate(["READ/SEARCH"], "non-specific", "internal") == Decision.ALLOW

    def test_single_action_deny(self):
        rt = RuleTable({"DELETE": Decision.DENY})
        res = rt.evaluate_detailed(["DELETE"], "non-specific", "internal")
        assert res.decision == Decision.DENY
        assert res.triggers == ["DELETE"]
        assert res.mixed is False

    def test_mixed_envelope_downgrades_deny_to_ask(self):
        # Tool can READ and DELETE; only DELETE denied. A blanket deny would
        # brick the benign READ use → downgrade to ASK, flag mixed.
        rt = RuleTable({"READ/SEARCH": Decision.ALLOW, "DELETE": Decision.DENY})
        res = rt.evaluate_detailed(["READ/SEARCH", "DELETE"], "non-specific", "internal")
        assert res.decision == Decision.ASK
        assert res.triggers == ["DELETE"]
        assert res.mixed is True

    def test_all_actions_denied_stays_deny(self):
        rt = RuleTable({"DELETE": Decision.DENY, "SEND": Decision.DENY})
        res = rt.evaluate_detailed(["DELETE", "SEND"], "non-specific", "internal")
        assert res.decision == Decision.DENY
        assert set(res.triggers) == {"DELETE", "SEND"}
        assert res.mixed is False

    def test_ask_beats_allow(self):
        rt = RuleTable({"SEND": Decision.ASK})
        assert rt.evaluate(["SEND"], "non-specific", "external") == Decision.ASK

    def test_mixed_deny_and_ask(self):
        # SPEND denied, SEND merely ask → still a mixed envelope (SEND is
        # not denied), so the call asks rather than hard-blocking.
        rt = RuleTable({"SEND": Decision.ASK, "SPEND": Decision.DENY})
        res = rt.evaluate_detailed(["SEND", "SPEND"], "financial", "external")
        assert res.decision == Decision.ASK
        assert res.triggers == ["SPEND"]
        assert res.mixed is True

    def test_sensitivity_deny_pervades_whole_tool(self):
        # Deny via sensitivity applies to the tool as a whole — never
        # downgraded even though the action is not itself denied.
        rt = RuleTable({"FINANCIAL": Decision.DENY, "CREATE": Decision.ALLOW})
        res = rt.evaluate_detailed(["CREATE"], "financial", "internal")
        assert res.decision == Decision.DENY
        assert res.triggers == ["FINANCIAL"]
        assert res.mixed is False

    def test_externality_as_rule_key(self):
        rt = RuleTable({"EXTERNAL": Decision.ASK, "READ/SEARCH": Decision.ALLOW})
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
