"""End-to-end: client -> proxy subprocess -> upstream stub subprocess."""

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
async def proxied_session():
    # An in-test context manager rather than a pytest fixture: stdio_client's
    # cancel scopes must enter and exit in the same task, and pytest-asyncio
    # tears async-generator fixtures down in a different task.
    async with stdio_client(PROXY_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def test_tools_list_passes_through():
    async with proxied_session() as session:
        result = await session.list_tools()
        names = {tool.name for tool in result.tools}
        assert names == {"echo", "add"}
        echo = next(t for t in result.tools if t.name == "echo")
        assert echo.description == "Echo the input text back."
        assert "text" in echo.inputSchema["properties"]


async def test_tools_call_passes_through():
    async with proxied_session() as session:
        result = await session.call_tool("echo", {"text": "hello"})
        assert not result.isError
        assert result.content[0].text == "hello"

        result = await session.call_tool("add", {"a": 2, "b": 3})
        assert not result.isError
        assert result.content[0].text == "5"


async def test_upstream_error_passes_through():
    async with proxied_session() as session:
        result = await session.call_tool("add", {"a": "not-an-int", "b": 3})
        assert result.isError
