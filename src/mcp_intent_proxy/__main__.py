"""CLI entry point: mcp-intent-proxy [options] [--] <upstream command> [args...]"""

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
        "--no-classify",
        action="store_true",
        help="disable the LLM classifier (passthrough-only mode)",
    )
    parser.add_argument(
        "--no-server-context",
        action="store_true",
        help="exclude server name/description from classifier input (ablation)",
    )
    parser.add_argument(
        "--server-name",
        default="",
        help="upstream server name fed to the classifier",
    )
    parser.add_argument(
        "--server-description",
        default="",
        help="upstream server description fed to the classifier",
    )
    parser.add_argument(
        "--rules",
        default=None,
        help="path to rules.yaml policy file (default: ~/.mcp-intent-proxy/rules.yaml)",
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

    async def _run() -> None:
        await run_proxy(
            upstream[0],
            upstream[1:],
            enable_classifier=not ns.no_classify,
            include_server_context=not ns.no_server_context,
            server_name=ns.server_name,
            server_description=ns.server_description,
            rules_path=ns.rules,
        )

    anyio.run(_run)


if __name__ == "__main__":
    main()
