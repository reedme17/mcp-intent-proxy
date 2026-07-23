"""Flip-rate: how deterministic is the classifier on repeated identical calls?

Usage:
    uv run python -m eval.run_flip --gt /path/to/ground-truth-120.json \
        [--runs 10] [--selection eval/flip_selection.json]

Runs each selected tool N times through a FRESH classifier each call (cache
bypassed — a cached result would trivially never flip). Reports, per tool and
per dimension, whether the label set was stable across all N runs, and the
overall flip rate. Selection (hard vs obvious) is read from --selection so the
"flips concentrate on hard tools" contrast can be shown.

This measures internal consistency (classifier vs. itself), NOT accuracy
(classifier vs. GT) — so it needs no ground-truth labels for the tools.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from eval.dataset import load_eval_items
from eval.run_eval import _load_dotenv


def _canon_action(labels: list[str]) -> tuple[str, ...]:
    return tuple(sorted(labels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Classifier flip-rate over repeated runs")
    parser.add_argument("--gt", required=True)
    parser.add_argument("--corpus-dir", default=None)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--selection", default="eval/flip_selection.json")
    parser.add_argument("--out", default="eval/flip_results.json")
    ns = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    _load_dotenv(repo_root)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    sel = json.loads((repo_root / ns.selection).read_text())
    hard_ids = set(sel["hard"])
    obvious_ids = set(sel["obvious"])
    wanted = hard_ids | obvious_ids

    items, _ = load_eval_items(ns.gt, ns.corpus_dir)
    by_id = {it.tool_id: it for it in items}

    from mcp_intent_proxy.classifier import Classifier

    results = []
    for tool_id in sorted(wanted):
        item = by_id.get(tool_id)
        if item is None:
            continue
        group = "hard" if tool_id in hard_ids else "obvious"
        runs = []
        for r in range(ns.runs):
            # Fresh classifier each run => empty cache => a real LLM call.
            clf = Classifier(include_server_context=True)
            res = clf.classify(
                tool_name=item.tool_name,
                tool_description=item.description + "\nInput schema:\n" + item.schema_text,
                input_schema={},
                server_name=item.server,
                server_description="",
                param_constraint=item.param_constraint,
            )
            runs.append({
                "action": _canon_action(res.action),
                "sensitivity": res.sensitivity,
                "externality": res.externality,
            })
        # Stability: number of distinct values across N runs; >1 means it flipped.
        a_distinct = len({r["action"] for r in runs})
        s_distinct = len({r["sensitivity"] for r in runs})
        e_distinct = len({r["externality"] for r in runs})
        results.append({
            "tool": tool_id, "group": group, "runs": ns.runs,
            "action_distinct": a_distinct, "sensitivity_distinct": s_distinct,
            "externality_distinct": e_distinct,
            "action_flipped": a_distinct > 1,
            "sensitivity_flipped": s_distinct > 1,
            "externality_flipped": e_distinct > 1,
            "action_values": [list(r["action"]) for r in runs],
            "sensitivity_values": [r["sensitivity"] for r in runs],
            "externality_values": [r["externality"] for r in runs],
        })
        flags = "".join(c for c, k in [("A","action_flipped"),("S","sensitivity_flipped"),("E","externality_flipped")] if results[-1][k])
        print(f"  [{group:7}] {tool_id.split('::')[-1].strip()[:32]:32} flip={flags or 'none'} "
              f"(A:{a_distinct} S:{s_distinct} E:{e_distinct} distinct/{ns.runs})")

    # Aggregate
    def rate(grp, dim):
        rows = [r for r in results if r["group"] == grp]
        if not rows:
            return None
        return sum(r[f"{dim}_flipped"] for r in rows) / len(rows)

    print("\n" + "=" * 60)
    print(f"FLIP RATE (fraction of tools that flipped over {ns.runs} runs)")
    print("=" * 60)
    for grp in ("hard", "obvious"):
        n = sum(1 for r in results if r["group"] == grp)
        print(f"  {grp:8} (n={n}): "
              f"Action {rate(grp,'action'):.0%}  "
              f"Sensitivity {rate(grp,'sensitivity'):.0%}  "
              f"Externality {rate(grp,'externality'):.0%}")

    (repo_root / ns.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {ns.out}")


if __name__ == "__main__":
    main()
