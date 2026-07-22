"""Transparent stdio proxy between an MCP client and one upstream MCP server.

The proxy speaks MCP on stdin/stdout (server side) and spawns the upstream
server as a subprocess (client side). tools/list and tools/call are forwarded
verbatim; raw request handlers are registered instead of the SDK's decorator
wrappers so upstream results — including isError and structuredContent — pass
through unmodified.

Every tools/call interception is recorded in a JSONL decision log.
"""

from __future__ import annotations

import os

import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from .logging import DecisionLog

SERVER_NAME = "mcp-intent-proxy"


def build_server(upstream: ClientSession, log: DecisionLog) -> Server:
    """Build a Server whose tool endpoints forward to the upstream session."""
    server = Server(SERVER_NAME)

    async def _list_tools(request: types.ListToolsRequest) -> types.ServerResult:
        cursor = request.params.cursor if request.params else None
        result = await upstream.list_tools(cursor=cursor)
        return types.ServerResult(result)

    async def _call_tool(request: types.CallToolRequest) -> types.ServerResult:
        name = request.params.name
        arguments = request.params.arguments or {}
        log.record(tool=name, arguments=arguments, decision="forward")
        result = await upstream.call_tool(name, arguments)
        return types.ServerResult(result)

    server.request_handlers[types.ListToolsRequest] = _list_tools
    server.request_handlers[types.CallToolRequest] = _call_tool
    return server


async def run_proxy(command: str, args: list[str]) -> None:
    """Spawn the upstream server and serve the proxy over stdio until EOF."""
    # Inherit the proxy's full environment: the SDK default is a minimal env,
    # which would strip variables the upstream server needs (PATH tweaks,
    # registry overrides, API keys).
    params = StdioServerParameters(command=command, args=args, env=dict(os.environ))
    log = DecisionLog()
    try:
        async with stdio_client(params) as (upstream_read, upstream_write):
            async with ClientSession(upstream_read, upstream_write) as upstream:
                await upstream.initialize()
                server = build_server(upstream, log)
                async with stdio_server() as (read_stream, write_stream):
                    await server.run(
                        read_stream,
                        write_stream,
                        server.create_initialization_options(),
                    )
    finally:
        log.close()
