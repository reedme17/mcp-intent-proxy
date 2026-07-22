# Design Notes

Decisions, limitations, and implications that shape this prototype.

## Defaults are fail-closed: unregulated categories ask

An action category with no rule in `rules.yaml` evaluates to **ask**, not
allow. A fresh install therefore protects from the first call: the user has
never stated a preference for, say, DELETE operations, so the proxy asks
rather than silently allowing.

Sensitivity and externality labels are *modifiers*: they participate in
evaluation only when an explicit rule names them (e.g. `FINANCIAL: deny`).
Every tool carries some sensitivity/externality value, so defaulting these
to ask would make every call interrupt regardless of action rules.

## Rules crystallize through use, not through configuration

The ask prompt is a three-way choice per category:

| Answer | This call | Rule written | Scope |
|---|---|---|---|
| Allow this call only | forwarded | none (session memory) | this tool, this session |
| Always allow category | forwarded | `CATEGORY: allow` | all tools of that category, forever |
| Never allow category | blocked | `CATEGORY: deny` | all tools of that category, forever |
| (no answer / cancel) | blocked | none | asked again next time |

Users never edit YAML. The rule table grows as questions get answered; each
settled category never asks again. Total interruptions are bounded by the
number of categories (10 actions + explicitly-ruled modifiers), not by the
number of tools — the core scaling advantage over per-tool authorization.

Asking is **per category, not per tool-call**: a tool classified as
[READ/SEARCH, SEND] with neither category settled produces one question for
READ/SEARCH and one for SEND. Every written rule is attributable to an
explicit user statement about exactly that category — consent is never
bundled. A "never" answer short-circuits: the call is already dead, so
remaining categories stay unsettled rather than badgering the user about a
blocked call.

The asymmetry between allow-once (no rule) and never (permanent rule) is
deliberate: a single yes is not blanket permission, but the "always allow"
option exists for users who *want* to settle a category — that is an
explicit, informed generalization, unlike inferring blanket consent from
one approval.

## Mixed envelopes: deny downgrades to ask, with a limitation

A tool's intent classification is a **capability envelope**: a SQL tool
accepting arbitrary queries can READ and DELETE; the classifier tags both.
The envelope answers "what could this tool do", not "what is this specific
call doing".

When a denied capability and a permitted capability share one envelope
(user denied DELETE but not READ/SEARCH, and the tool has both), a hard
deny would brick the tool's benign majority use. Instead the proxy
downgrades to **ask**, telling the user which capability triggered the
question. Two denials never downgrade: a deny via sensitivity/externality
rule (e.g. `FINANCIAL: deny`) pervades the whole tool, and a tool whose
*every* action is denied has no benign remainder to protect.

**Limitation.** Per-call arbitration is a human patch over a granularity
gap. The proxy decides at the tool level but the risk lives at the call
level: `SELECT * FROM orders` and `DROP TABLE orders` are the same tool,
the same envelope, the same ask. Argument-level classification (parsing the
SQL, inspecting the arguments) would close the gap but moves the classifier
onto attacker-influenced input with a much larger surface — argument values
are the primary channel for prompt injection — and defeats
classify-once-per-signature caching, since arguments differ per call.

**Implication for tool design.** Intent-scoped authorization structurally
favors narrow tools. A `read_table` / `run_migration` pair is enforceable
cleanly; a swiss-army `execute_sql` forces either over-blocking, per-call
interruption, or argument inspection. If intent-level permissioning becomes
common, MCP server authors have an incentive to expose capability-scoped
tools rather than envelope-maximal ones — the same pressure OAuth scopes
put on API design a decade earlier.

## Deny messages are agent prompts

The deny result's text is read by the calling agent, not directly by the
user. It is written as an instruction: what was blocked, why (which
category), that retrying or reaching the same goal via other tools is
prohibited, and — for mixed envelopes — that a narrower tool without the
denied capability is the legitimate path. The agent relays the explanation;
the proxy supplies the script.

## Late-registered tools

A tool absent from the initial `tools/list` that appears later (rug-pull
pattern) is re-synced, classified, and escalated to ask-with-warning even
if rules would allow it. An explicit deny rule still wins — escalation only
replaces silent allows, never overrides denials.
