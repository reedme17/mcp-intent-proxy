# Decision Flow

The state machine for a single `tools/call`, from the agent's request to a
forwarded result or a blocked error. Mirrors `proxy.py::_call_tool`.

```
==============================================================================
 START: Agent calls a tool
 Example: agent calls "send_email" with {to: "bob@x.com", body: "hi"}
==============================================================================
                                   |
                                   v
+----------------------------------------------------------------------------+
| S1. CLASSIFIER ENABLED?                                                    |
| Was the proxy started without --no-classify? (default: enabled)           |
+-----------------------------+----------------------------+-----------------+
                          YES  |                       NO   |
                               |                            v
                               |     +-------------------------------------+
                               |     | EXIT: FORWARD (passthrough)         |
                               |     | No policy check. Pass to upstream,  |
                               |     | return result unchanged.            |
                               |     +-------------------------------------+
                               v
+----------------------------------------------------------------------------+
| S2. TOOL IN REGISTRY?                                                      |
| Registry = tools seen in a prior tools/list (pre-classified at startup).  |
| Example: "send_email" was listed at startup -> in registry                |
+-----------------------------+----------------------------+-----------------+
                          YES  |                        NO  |
                               |                            v
                               |     +-------------------------------------+
                               |     | S3. RE-SYNC                         |
                               |     | Pull a fresh tools/list from        |
                               |     | upstream (server may have added it).|
                               |     +--------+-------------------+--------+
                               |         FOUND|             STILL |
                               |              |           MISSING |
                               |              |                   v
                               |              |  +-------------------------------+
                               |              |  | EXIT: DENY (nonexistent)      |
                               |              |  | Tool is not on the server     |
                               |              |  | (agent hallucinated the name).|
                               |              |  | "Do not retry. Inform user."  |
                               |              |  +-------------------------------+
                               |              v
                               |     +-------------------------------------+
                               |     | FLAG late_registered = true         |
                               |     | Tool appeared AFTER startup.        |
                               |     +--------+----------------------------+
                               |              |
                               v              v
+----------------------------------------------------------------------------+
| S4. CLASSIFY (cache hit from pre-classify, or LLM call if late)           |
| Output: Action[] + Sensitivity + Externality + confidence + rationale     |
| Example: send_email -> Action=[SEND], Sens=personal-communications,       |
|          Ext=external, conf=0.92                                          |
| (Fail-closed inside classifier: unparseable/low-confidence -> restrictive |
|  fallback; transient errors are NOT cached.)                              |
+-------------------------------------+--------------------------------------+
                                      v
+----------------------------------------------------------------------------+
| S5. EVALUATE RULES  ->  EvalResult(decision, triggers, mixed)             |
| Per-label: explicit rule, or ASK default for unregulated ACTION labels.   |
| Sensitivity/externality participate only via explicit rules.              |
| Most restrictive wins (deny > ask > allow).                               |
|                                                                            |
| Example: rules={SEND: ask}          -> ASK, triggers=[SEND], mixed=false  |
| Example: rules={} (empty)           -> ASK, triggers=[SEND] (fail-closed) |
| Example: rules={DELETE: deny},                                            |
|          tool=[READ/SEARCH,DELETE]  -> ASK, triggers=[DELETE], mixed=TRUE |
|          (mixed envelope: deny would brick the benign READ use)           |
| Example: rules={DELETE: deny},                                            |
|          tool=[DELETE]              -> DENY, triggers=[DELETE]            |
| Example: rules={FINANCIAL: deny}    -> DENY (sensitivity pervades tool)   |
+-------------------------------------+--------------------------------------+
                                      v
+----------------------------------------------------------------------------+
| S6. LATE-REGISTERED OVERRIDE                                              |
| If decision==ALLOW AND late_registered:  ALLOW -> ASK (+ warning).        |
| DENY and ASK are already strict enough; not overridden.                   |
| Example: a late "read_passwords" that rules would ALLOW is escalated to   |
|          ASK so the user sees the suspicious late addition.               |
+-------------------------------------+--------------------------------------+
                                      v
+----------------------------------------------------------------------------+
| S7. EXECUTE DECISION                                                      |
+--------+--------------------------+--------------------------------+-------+
      ALLOW                       DENY                             ASK
         |                          |                               |
         v                          v                               v
+----------------+  +-----------------------------+  +----------------------------+
| EXIT: FORWARD  |  | EXIT: DENY (policy)         |  | S8. SESSION MEMORY         |
| Call upstream, |  | isError + instructive msg:  |  | Did the user allow-once    |
| return result. |  | "Blocked by policy.         |  | this exact tool earlier    |
| Example:       |  |  intent=DELETE. Do not      |  | this session?              |
| read_file runs |  |  retry. Do not reach the    |  +------+---------------+-----+
| normally.      |  |  same goal via other tools. |     YES |            NO |
+----------------+  |  Inform user; they can edit |         |              |
                    |  rules.yaml."               |         v              |
                    | Agent relays this to user.  |  +----------------+    |
                    +-----------------------------+  | EXIT: FORWARD  |    |
                                                     | (skip re-ask)  |    |
                                                     +----------------+    |
                                                                          v
+----------------------------------------------------------------------------+
| S9. ASK USER  --  one elicitation PER trigger category                    |
| (trigger labels = the categories that produced the ASK; falls back to the |
|  tool's full Action[] when nothing was regulated yet)                     |
|                                                                            |
| For each category, show a 3-way choice:                                   |
|   "Agent wants 'send_email' (SEND, personal-communications, external).    |
|    [Allow this call only] [Always allow SEND] [Never allow SEND]"         |
|   (mixed envelope adds: "combines denied + permitted capabilities")       |
|   (late-registered adds: "this tool was not present at startup")          |
|                                                                            |
| A "Never" answer short-circuits the loop (call is already dead).          |
+----+-----------------------+-----------------------+----------------------+
     |                       |                       |                      |
 ALLOW_ONCE           ALWAYS_CATEGORY         NEVER_CATEGORY          CANCEL / no answer
     |                       |                       |                      |
     v                       v                       v                      v
+-----------+   +----------------------+  +----------------------+  +------------------+
| Forward   |   | Write CATEGORY:allow |  | Write CATEGORY:deny  |  | Block this call  |
| this call |   | Forward this call    |  | Block this call      |  | Write NO rule    |
| Remember  |   | (persistent, explicit|  | Future tools of this |  | Ask again next   |
| for session|  |  opt-in to the       |  | category auto-denied,|  | time             |
| Write NO  |   |  category)           |  | incl. not-yet-       |  |                  |
| rule      |   |                      |  | installed ones       |  | Silence is not   |
|           |   |                      |  | ("deny once, cover   |  | an opinion.      |
| One yes = |   |                      |  |  everywhere")        |  |                  |
| not a     |   |                      |  |                      |  |                  |
| category  |   |                      |  | (mixed envelope adds |  |                  |
| opt-in    |   |                      |  |  "use a narrower     |  |                  |
|           |   |                      |  |  tool" to the msg)   |  |                  |
+-----------+   +----------------------+  +----------------------+  +------------------+
```

## Why the answers are asymmetric

| Answer | This call | Rule written | Scope of effect |
|---|---|---|---|
| Allow this call only | forwarded | none (session memory) | this tool, this session |
| Always allow category | forwarded | `CATEGORY: allow` | all tools of category, forever |
| Never allow category | blocked | `CATEGORY: deny` | all tools of category, forever |
| Cancel / no answer | blocked | none | re-asked next time |

- **One yes is not blanket permission.** Allow-once writes no rule — the
  user might only need it this once. "Always" exists for users who *want*
  to settle the category; that is an explicit, informed generalization.
- **One no generalizes.** A clear "never" is a durable preference: the user
  does not want to be bothered by this category again. This is the core
  "deny once, cover everywhere" mechanism.
- **Silence generalizes to nothing.** No answer is not a decision, so no
  rule is written and the question returns next time.

## Terminal states

| Exit | Result to agent | Rule side effect |
|---|---|---|
| Passthrough (S1) | forwarded, unchanged | — |
| Forward (S7 allow / S8 remembered / S9 allow) | upstream result | allow rule only on "always" |
| Deny — nonexistent (S3) | isError, "tool not on server" | none |
| Deny — policy (S7) | isError, instructive message | — (rule already existed) |
| Deny — user "never" (S9) | isError, instructive message | `CATEGORY: deny` written |
| Deny — cancelled (S9) | isError, "awaiting confirmation" | none |
```
