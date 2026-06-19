# Build Kickoff

Starting point and minimal-MVP plan for the intent-based MCP authorization proxy.

## Goal

Build an intent↔tool semantic mapping engine:

1. Intercept an agent's MCP `tools/list` and `tools/call`.
2. Use an LLM to classify each call from `(tool name + description + arguments)` into an intent category.
3. Check the intent against a rule table → `allow` / `deny` / `ask`.
4. Support **single-deny generalization**: one deny becomes a standing, category-wide rule covering every tool sharing that intent — including tools never seen before.
5. On deny, return a tool result with `isError: true` and a message like `blocked by policy: intent=money`.

## Constraints

- **Scope**: stdio transport + a single upstream MCP server. No HTTP/SSE, no multi-server (future work).
- **Classifier**: LLM at temperature 0. **Cache by tool signature** — intent is a property of `(tool, argument-shape)`, not of each individual call, so classify once per signature and reuse. Optionally pre-classify everything at `tools/list` time.
- **Failure policy**: fail-closed. Low confidence → default to `ask` or `deny`.
- **Threat awareness**: the LLM sits on the authorization path, so a tool's name/description/arguments are attacker-influenced text. A malicious tool may rename or re-describe itself to be classified as low-risk, or embed prompt-injection aimed at the classifier. Treat misclassification cost as asymmetric (classifying a destructive call as safe is the dangerous direction). Log decisions; measure determinism (same call, repeated runs).

## What to fork (don't rebuild the proxy)

The MCP transport/proxy plumbing is a solved problem. Candidates to build on:
- `sparfenyuk/mcp-proxy` — a stdio transport bridge; fork it and add the interception hook for full control.
- Docker MCP Gateway — has an interceptor hook (`before` tools/call); the classifier can be a small HTTP service, with near-zero proxy code.
- Microsoft's MCP security gateway reference — useful architectural reference (intercept → scan → approval callback → fail-closed).

The only genuinely new component to write is the **LLM intent-classification decision layer**.

## First milestone (target: a weekend)

1. Stand up a stdio proxy in front of one real MCP server (e.g. a filesystem server), pass calls through unchanged — confirm the agent still works.
2. Intercept `tools/call`: log the tool name + arguments before forwarding.
3. Add the LLM classifier: prompt = tool name + description + arguments → one intent label. Temperature 0. Cache by tool signature.
4. Add a YAML rule table (`allow` / `deny` / `ask` per intent) and enforce it; on deny, return `isError` with the policy message.
5. Implement single-deny generalization: a deny on one tool writes a category-level rule that then applies to other tools of the same intent.

## Test inputs

A corpus of ~510 real MCP tool schemas (verbatim name + description + input schema, across ~28 servers) is available as classifier test/evaluation input. Use it to sanity-check the classifier: e.g. a "read-only" SQL tool whose only argument is free-form SQL, a calendar "create event" tool that can email external attendees, an explicit "delete" tool. (Provided separately — ask for the path.)

## First question to resolve

Fork `sparfenyuk/mcp-proxy` (full code control) or use the Docker MCP Gateway interceptor (less code)? Pick one, then start from milestone step 1.
