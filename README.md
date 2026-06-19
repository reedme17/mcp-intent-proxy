# mcp-intent-proxy

A research prototype: an **intent-based authorization layer for AI agents**. It sits between an AI agent and the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) tools it calls, classifies each tool call into a human-legible **intent category** (spend money, send outbound, delete, touch personal data, …), checks it against user-declared rules, and allows / denies / asks — supporting **single-deny → cross-tool generalization** ("deny once, cover everywhere").

This is a research prototype, not a production tool.

## What it does

```
AI agent  →  [ this proxy ]  →  real MCP tools
                ├─ intercept tools/list and tools/call
                ├─ LLM classifier: (tool name + description + args) → intent category
                ├─ rule table: allow / deny / ask  (+ single-deny generalizes across tools)
                └─ deny → return isError + "blocked by policy: intent=<X>"
```

## Scope (prototype)

- **Transport**: stdio + a single upstream server (HTTP/SSE and multi-server are future work).
- **Classifier**: LLM, temperature 0, **cached by tool signature** (intent is a property of `(tool, argument-shape)`, not of each call).
- **Failure policy**: fail-closed (low confidence → ask/deny).

## Status

🚧 Early prototype. See `KICKOFF.md` for the build plan and starting point.

## Design docs

The research design lives in a separate (private) research repo. Key references for building:
- engine feasibility analysis (what to fork, latency strategy, threat model)
- system section §4 and threat model §2.2 of the paper draft
- a corpus of ~510 real MCP tool schemas used as classifier test inputs

## License

TBD (intend to open-source on publication).
