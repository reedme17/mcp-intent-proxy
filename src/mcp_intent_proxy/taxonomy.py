"""Intent taxonomy: the fixed category system for tool classification.

Locked taxonomy verified at IRR >= 0.88 across 3 independent annotators.
"""

from __future__ import annotations

from enum import Enum

# --- Action (multi-select) ---

ACTIONS: list[str] = [
    "READ/SEARCH",
    "CREATE",
    "MODIFY/MANAGE",
    "DELETE",
    "SEND",
    "SPEND",
    "EXECUTE",
    "PHYSICAL",
    "CREDENTIAL/IDENTITY",
    "OTHER",
]

# --- Sensitivity (single-select) ---

SENSITIVITIES: list[str] = [
    "health",
    "identity-credential",
    "financial",
    "personal-communications",
    "location",
    "non-specific",
]

# --- Externality (single-select) ---

EXTERNALITIES: list[str] = [
    "internal",
    "external",
]


class IntentResult:
    """Validated classification result for a single tool."""

    __slots__ = ("action", "sensitivity", "externality", "confidence", "rationale")

    def __init__(
        self,
        *,
        action: list[str],
        sensitivity: str,
        externality: str,
        confidence: float = 1.0,
        rationale: str = "",
    ) -> None:
        self.action = action
        self.sensitivity = sensitivity
        self.externality = externality
        self.confidence = confidence
        self.rationale = rationale

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "sensitivity": self.sensitivity,
            "externality": self.externality,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


# The fail-closed fallback: maximally restrictive classification used when
# the LLM response is unparseable or confidence is below threshold.
FAIL_CLOSED_RESULT = IntentResult(
    action=["OTHER"],
    sensitivity="non-specific",
    externality="external",
    confidence=0.0,
    rationale="Classification failed or confidence below threshold; fail-closed.",
)


def validate_action(values: list[str]) -> list[str]:
    """Return only valid action labels; empty list if none valid."""
    return [v for v in values if v in ACTIONS]


def validate_sensitivity(value: str) -> str | None:
    """Return the value if valid, else None."""
    return value if value in SENSITIVITIES else None


def validate_externality(value: str) -> str | None:
    """Return the value if valid, else None."""
    return value if value in EXTERNALITIES else None
