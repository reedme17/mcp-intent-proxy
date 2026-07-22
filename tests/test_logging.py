"""Verify that tools/call interceptions are recorded in the decision log."""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

UPSTREAM = str(Path(__file__).parent / "upstream_stub.py")

PROXY_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "mcp_intent_proxy", "--no-classify", "--", sys.executable, UPSTREAM],
)


@asynccontextmanager
async def proxied_session(tmp_path: Path):
    import os

    env = dict(os.environ)
    env["MCP_INTENT_PROXY_LOG_DIR"] = str(tmp_path)
    params = StdioServerParameters(
        command=PROXY_PARAMS.command,
        args=list(PROXY_PARAMS.args or []),
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def test_decisions_logged(tmp_path: Path):
    async with proxied_session(tmp_path) as session:
        await session.call_tool("echo", {"text": "hi"})
        await session.call_tool("add", {"a": 1, "b": 2})

    log_file = tmp_path / "decisions.jsonl"
    assert log_file.exists()
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert len(lines) == 2

    assert lines[0]["tool"] == "echo"
    assert lines[0]["arguments"] == {"text": "hi"}
    assert lines[0]["decision"] == "forward"
    assert "ts" in lines[0]

    assert lines[1]["tool"] == "add"
    assert lines[1]["arguments"] == {"a": 1, "b": 2}
