"""Structured decision log for the proxy.

Logs are written as one JSON object per line to a file (default:
~/.mcp-intent-proxy/decisions.jsonl). Each entry captures a tools/call
interception: tool name, arguments, and — once the classifier and rule
engine are wired in — intent classification and policy decision.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_LOG_DIR = Path.home() / ".mcp-intent-proxy"
ENV_LOG_DIR = "MCP_INTENT_PROXY_LOG_DIR"

logger = logging.getLogger("mcp_intent_proxy")


def setup_stderr_logging() -> None:
    """Send human-readable log lines to stderr (visible in debug mode)."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def _resolve_log_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get(ENV_LOG_DIR)
    if env:
        return Path(env)
    return DEFAULT_LOG_DIR


class DecisionLog:
    """Append-only JSONL log for tool-call decisions."""

    def __init__(self, log_dir: Path | None = None) -> None:
        self._dir = _resolve_log_dir(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "decisions.jsonl"
        self._file = open(self._path, "a", encoding="utf-8")

    def record(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        decision: str = "forward",
        intent: dict[str, Any] | None = None,
        rule: str | None = None,
        flags: list[str] | None = None,
    ) -> None:
        entry = {
            "ts": time.time(),
            "tool": tool,
            "arguments": arguments,
            "decision": decision,
        }
        if intent is not None:
            entry["intent"] = intent
        if rule is not None:
            entry["rule"] = rule
        if flags:
            entry["flags"] = flags
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        self._file.write(line + "\n")
        self._file.flush()
        logger.debug(
            "tool=%s decision=%s intent=%s", tool, decision, intent
        )

    def close(self) -> None:
        self._file.close()
