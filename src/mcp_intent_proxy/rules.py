"""Rule engine: load a YAML policy file and evaluate intent against it.

A rule table maps intent categories to decisions (allow / deny / ask).
Rules are checked against the classified Action labels — the most restrictive
matching rule wins (deny > ask > allow).

Defaults are fail-closed: an action category with no rule evaluates to ASK
(the user has never stated a preference, so ask). Sensitivity/externality
labels are modifiers — they only participate when an explicit rule names them.

Mixed envelopes: when a tool's action envelope contains BOTH denied and
non-denied capabilities (e.g. a SQL tool that can READ and DELETE, with only
DELETE denied), a hard deny would brick the tool's benign uses. Such tools
downgrade DENY to ASK — per-call human arbitration instead of a blanket block.
A tool whose every action is denied (or denied via a sensitivity/externality
rule, which pervades the whole tool) stays DENY.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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

# Fail-closed: unregulated action categories ask instead of silently allowing.
DEFAULT_DECISION = Decision.ASK

DEFAULT_RULES_PATH = Path.home() / ".mcp-intent-proxy" / "rules.yaml"


@dataclass
class EvalResult:
    """Outcome of a rule evaluation.

    triggers: the category labels whose rule (or default) produced the final
    decision — the labels the user is being asked about, and the only labels
    that rule-writing generalization may touch.
    mixed: True when a DENY was downgraded to ASK because the action envelope
    also contains non-denied capabilities.
    """

    decision: Decision
    triggers: list[str] = field(default_factory=list)
    mixed: bool = False


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
        """Check classified intent against the rule table (decision only)."""
        return self.evaluate_detailed(actions, sensitivity, externality).decision

    def evaluate_detailed(
        self, actions: list[str], sensitivity: str, externality: str
    ) -> EvalResult:
        """Evaluate intent against the rules; report triggers and mixed status.

        Per-label decisions:
        - Action labels: explicit rule, or DEFAULT_DECISION (ask) when
          unregulated — the user has never stated a preference for that
          category, so ask rather than silently allow.
        - Sensitivity/externality: participate only via explicit rules.
          They are always present on every tool, so defaulting them to ask
          would make every call ask regardless of action rules.

        Mixed-envelope downgrade: DENY becomes ASK when the deny comes from
        action rules that cover only part of the tool's action envelope
        (some actions are not denied). A blanket block would brick the
        tool's benign capabilities, so a human arbitrates per call instead.
        Denies via sensitivity/externality rules pervade the whole tool and
        are never downgraded; neither is a tool whose every action is denied.
        """
        per_label: dict[str, Decision] = {}
        action_keys: list[str] = []
        for a in actions:
            key = a.upper()
            action_keys.append(key)
            per_label[key] = self._rules.get(key, DEFAULT_DECISION)
        for modifier in (sensitivity, externality):
            key = modifier.upper()
            if key in self._rules:
                per_label[key] = self._rules[key]

        if not per_label:
            return EvalResult(Decision.ALLOW)

        worst = max(per_label.values(), key=lambda d: _SEVERITY[d])

        if worst == Decision.ALLOW:
            return EvalResult(Decision.ALLOW)

        if worst == Decision.DENY:
            denied = [k for k, d in per_label.items() if d == Decision.DENY]
            denied_modifiers = [k for k in denied if k not in action_keys]
            nondenied_actions = [
                k for k in action_keys if per_label[k] != Decision.DENY
            ]
            if denied_modifiers or not nondenied_actions:
                return EvalResult(Decision.DENY, denied)
            return EvalResult(Decision.ASK, denied, mixed=True)

        triggers = [k for k, d in per_label.items() if d == Decision.ASK]
        return EvalResult(Decision.ASK, triggers)

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
