"""Rule engine: load a YAML policy file and evaluate intent against it.

A rule table maps intent categories to decisions (allow / deny / ask).
Rules are checked against the classified Action labels — the most restrictive
matching rule wins (deny > ask > allow).
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("mcp_intent_proxy.rules")


class Decision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


# Severity ordering for "most restrictive wins" logic.
_SEVERITY = {Decision.ALLOW: 0, Decision.ASK: 1, Decision.DENY: 2}

DEFAULT_DECISION = Decision.ALLOW

DEFAULT_RULES_PATH = Path.home() / ".mcp-intent-proxy" / "rules.yaml"


class RuleTable:
    """Intent-category rule table loaded from YAML."""

    def __init__(self, rules: dict[str, Decision] | None = None) -> None:
        self._rules: dict[str, Decision] = {
            k.upper(): v for k, v in (rules or {}).items()
        }

    @classmethod
    def load(cls, path: Path | None = None) -> "RuleTable":
        """Load rules from a YAML file. Missing file → empty table (all allow)."""
        import yaml

        target = path or DEFAULT_RULES_PATH
        if not target.exists():
            logger.info("No rules file at %s, defaulting to allow-all", target)
            return cls()

        with open(target, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        rules: dict[str, Decision] = {}
        for category, decision_str in raw.items():
            category = str(category).upper()
            try:
                rules[category] = Decision(str(decision_str).lower())
            except ValueError:
                logger.warning(
                    "Invalid decision '%s' for category '%s', ignoring",
                    decision_str,
                    category,
                )
        return cls(rules)

    def evaluate(self, actions: list[str], sensitivity: str, externality: str) -> Decision:
        """Check classified intent against the rule table.

        The most restrictive matching rule wins across all action labels,
        sensitivity, and externality. Unmatched categories default to allow.
        """
        decision, _ = self.evaluate_detailed(actions, sensitivity, externality)
        return decision

    def evaluate_detailed(
        self, actions: list[str], sensitivity: str, externality: str
    ) -> tuple[Decision, list[str]]:
        """Evaluate and also report which labels produced the final decision.

        Returns (decision, trigger_labels). trigger_labels are the category
        keys whose rule matches the final (most restrictive) decision — the
        labels the user actually objected to. Single-deny generalization must
        generalize only these, never every label on the tool: a tool tagged
        [READ/SEARCH, SEND] denied because of a SEND rule says nothing about
        the user's stance on READ/SEARCH.
        """
        matched: dict[str, Decision] = {}
        worst = DEFAULT_DECISION
        for label in [*actions, sensitivity, externality]:
            key = label.upper()
            decision = self._rules.get(key, DEFAULT_DECISION)
            matched[key] = decision
            if _SEVERITY[decision] > _SEVERITY[worst]:
                worst = decision
        if worst == DEFAULT_DECISION:
            return worst, []
        triggers = [k for k, d in matched.items() if d == worst]
        return worst, triggers

    def set_rule(self, category: str, decision: Decision) -> None:
        """Programmatically add or update a rule."""
        self._rules[category.upper()] = decision

    def save(self, path: Path | None = None) -> None:
        """Persist the current rules to YAML."""
        import yaml

        target = path or DEFAULT_RULES_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.value for k, v in sorted(self._rules.items())}
        with open(target, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @property
    def rules(self) -> dict[str, Decision]:
        return dict(self._rules)

    def __repr__(self) -> str:
        return f"RuleTable({self._rules})"
