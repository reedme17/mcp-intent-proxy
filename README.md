# mcp-intent-proxy

A research prototype: an **intent-based authorization layer for AI agents**. It sits between an AI agent and the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) tools it calls, classifies each tool call into a human-legible **intent category** (spend money, send outbound, delete, touch personal data, …), checks it against user-declared rules, and allows / denies / asks — supporting **single-deny → cross-tool generalization** ("deny once, cover everywhere").

This is a research prototype, not a production tool.

## What it does

```
AI agent  →  [ this proxy ]  →  real MCP tools
                ├─ intercept tools/list: pre-classify every tool (zero latency at call time)
                ├─ LLM classifier: tool schema (+ server context) → Action[] + Sensitivity + Externality
                ├─ rule table: allow / deny / ask per category — unregulated categories ask
                ├─ ask → user chooses: this call only / always allow category / never allow category
                ├─ "never" generalizes: one deny covers every tool of that category, incl. future ones
                └─ deny → isError with an instructive message for the agent
```

Rules crystallize through use: users answer questions, never edit YAML.
Interruptions are bounded by the number of categories, not tools.
See `docs/design-notes.md` for design decisions and limitations.

## Scope (prototype)

- **Transport**: stdio + a single upstream server (HTTP/SSE and multi-server are future work).
- **Classifier**: LLM, temperature 0, **cached by tool signature** (intent is a property of `(tool, argument-shape)`, not of each call).
- **Failure policy**: fail-closed — unparseable/low-confidence classifications deny; unregulated categories ask; unknown tools re-sync then deny; late-registered tools escalate to ask-with-warning.

## Usage

```bash
uv sync
export ANTHROPIC_API_KEY=sk-...
mcp-intent-proxy [--rules path/to/rules.yaml] [--no-server-context] -- <upstream server command>
```

Flags: `--no-classify` (passthrough only), `--no-server-context` (ablation:
exclude server identity from classifier input), `--rules` (default
`~/.mcp-intent-proxy/rules.yaml`). Decision log: `~/.mcp-intent-proxy/decisions.jsonl`.

## Status

🚧 Working prototype: proxy, classifier, rules, ask flow, and generalization
are implemented and tested. Evaluation harness and baselines are next.

## Design docs

The research design lives in a separate (private) research repo. Key references for building:
- engine feasibility analysis (what to fork, latency strategy, threat model)
- system section §4 and threat model §2.2 of the paper draft
- a corpus of ~510 real MCP tool schemas used as classifier test inputs

## License

TBD (intend to open-source on publication).
