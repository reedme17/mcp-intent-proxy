"""Signature-based cache for tool classifications.

Intent is a property of (tool name, description, input schema) — not of each
individual call's argument values. We cache by a stable hash of the tool's
schema so that repeated calls to the same tool never re-invoke the LLM.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .taxonomy import IntentResult


def tool_signature(name: str, description: str, input_schema: dict[str, Any]) -> str:
    """Compute a stable cache key for a tool's identity."""
    blob = json.dumps(
        {"name": name, "description": description, "schema": input_schema},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class ClassificationCache:
    """In-memory cache mapping tool signatures to IntentResults."""

    def __init__(self) -> None:
        self._store: dict[str, IntentResult] = {}

    def get(self, sig: str) -> IntentResult | None:
        return self._store.get(sig)

    def put(self, sig: str, result: IntentResult) -> None:
        self._store[sig] = result

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, sig: str) -> bool:
        return sig in self._store
