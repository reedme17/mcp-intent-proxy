"""Minimal upstream MCP server used as the proxy's target in tests."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("upstream-stub")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the input text back."""
    return text


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    mcp.run()
