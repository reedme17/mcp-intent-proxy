"""CLI entry point: mcp-intent-proxy [--] <upstream command> [args...]"""

from __future__ import annotations

import argparse

import anyio

from .proxy import run_proxy


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mcp-intent-proxy",
        description=(
            "Run an MCP stdio proxy in front of one upstream MCP server. "
            "Everything after the proxy's own options is the upstream command."
        ),
    )
    parser.add_argument(
        "upstream",
        nargs=argparse.REMAINDER,
        help="upstream MCP server command and its arguments",
    )
    ns = parser.parse_args()

    upstream = ns.upstream
    if upstream and upstream[0] == "--":
        upstream = upstream[1:]
    if not upstream:
        parser.error("missing upstream server command")

    anyio.run(run_proxy, upstream[0], upstream[1:])


if __name__ == "__main__":
    main()
