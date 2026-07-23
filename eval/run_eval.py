"""Evaluate the intent classifier against an adjudicated ground-truth corpus.

Usage:
    uv run python -m eval.run_eval --gt /path/to/ground-truth-120.json \
        [--corpus-dir /path/to/schemas] [--no-server-context] [--limit N] \
        [--out results.json]

Reads ANTHROPIC_API_KEY from the environment or from a .env file in the repo
root. The ground-truth corpus is read from the path given; nothing is written
back to it. Results go to --out (default: eval/last_results.json, gitignored).

Metrics:
- Action: multi-label. Reported as exact-set match (strict) and Jaccard.
- Sensitivity / Externality: single-label exact match, over the subset not
  flagged excluded_S_E (matching how the human/LLM baseline was computed).
Confusion matrices are printed for sensitivity and externality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from eval.dataset import EvalItem, load_eval_items


def _load_dotenv(repo_root: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, no export, no quotes handling."""
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _main_action_match(pred: set[str], gt: set[str]) -> bool:
    """Lenient action agreement mirroring the ground-truth analysis rule.

    The corpus's 95.8% figure counts "multi-action over/under-selection" as a
    false disagreement (agreement): differences that are only a missing or
    extra secondary label, with no contradicting label, are treated as a match.
    Operationally: one label set is a subset of the other AND they overlap
    (neither is empty). A genuine contradiction — each set has a label the
    other lacks — is a real disagreement and does not match.
    """
    if not pred or not gt:
        return pred == gt
    if not (pred & gt):
        return False
    return pred <= gt or gt <= pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate intent classifier vs ground truth")
    parser.add_argument("--gt", required=True, help="path to ground-truth-120.json")
    parser.add_argument("--corpus-dir", default=None, help="dir with labeled-dataset-part*.md (default: gt's dir)")
    parser.add_argument("--no-server-context", action="store_true", help="ablation: exclude server context")
    parser.add_argument("--offset", type=int, default=0, help="skip the first OFFSET items (for held-out slices)")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only N items (after offset)")
    parser.add_argument("--out", default="eval/last_results.json", help="where to write detailed results")
    parser.add_argument("--dev-split", type=int, default=20, help="first N items are the dev (prompt-aligned) group; rest are held-out test")
    ns = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    _load_dotenv(repo_root)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set (env or .env). Cannot run classifier.")

    items, meta = load_eval_items(ns.gt, ns.corpus_dir)
    if ns.offset:
        items = items[ns.offset :]
    if ns.limit:
        items = items[: ns.limit]

    print(f"Loaded {meta['n_joined']}/{meta['n_entries']} items "
          f"({meta['n_missing']} unjoined). Evaluating {len(items)}.")
    print(f"Baseline (independent LLM vs human GT): {meta['agreement_stats']}")
    print(f"Server context: {'OFF (ablation)' if ns.no_server_context else 'ON'}")
    print()

    # Import here so a missing key fails fast above, before network setup.
    from mcp_intent_proxy.classifier import Classifier

    clf = Classifier(include_server_context=not ns.no_server_context)

    action_exact = 0
    action_main = 0
    action_jaccard_sum = 0.0
    sens_correct = 0
    sens_total = 0
    ext_correct = 0
    ext_total = 0
    sens_confusion: dict[tuple[str, str], int] = defaultdict(int)
    ext_confusion: dict[tuple[str, str], int] = defaultdict(int)
    failures: list[dict] = []
    details: list[dict] = []

    for idx, item in enumerate(items, 1):
        result = clf.classify(
            tool_name=item.tool_name,
            tool_description=item.description + "\nInput schema:\n" + item.schema_text,
            input_schema={},
            server_name=item.server,
            server_description="",
            param_constraint=item.param_constraint,
        )

        pred_actions = set(result.action)
        gt_actions = set(item.gt_action)
        exact = pred_actions == gt_actions
        main = _main_action_match(pred_actions, gt_actions)
        jac = _jaccard(pred_actions, gt_actions)
        action_exact += int(exact)
        action_main += int(main)
        action_jaccard_sum += jac

        # Sensitivity / externality only over non-excluded items.
        sens_ok = ext_ok = None
        if not item.excluded_s_e:
            sens_total += 1
            sens_ok = result.sensitivity == item.gt_sensitivity
            sens_correct += int(sens_ok)
            sens_confusion[(item.gt_sensitivity, result.sensitivity)] += 1

            ext_total += 1
            ext_ok = result.externality == item.gt_externality
            ext_correct += int(ext_ok)
            ext_confusion[(item.gt_externality, result.externality)] += 1

        if not exact or sens_ok is False or ext_ok is False:
            failures.append({
                "tool": item.tool_id,
                "gt_action": sorted(gt_actions), "pred_action": sorted(pred_actions),
                "gt_sensitivity": item.gt_sensitivity, "pred_sensitivity": result.sensitivity,
                "gt_externality": item.gt_externality, "pred_externality": result.externality,
                "confidence": result.confidence,
            })

        details.append({
            "tool": item.tool_id,
            "gt_action": sorted(gt_actions), "pred_action": sorted(pred_actions),
            "action_exact": exact, "action_main": main, "action_jaccard": round(jac, 3),
            "gt_sensitivity": item.gt_sensitivity, "pred_sensitivity": result.sensitivity,
            "gt_externality": item.gt_externality, "pred_externality": result.externality,
            "excluded_s_e": item.excluded_s_e, "confidence": result.confidence,
        })

        if idx % 10 == 0:
            print(f"  ...{idx}/{len(items)}")

    n = len(items)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Action   exact-set match:  {action_exact}/{n} = {action_exact/n:.1%}  (strict: sets identical)")
    print(f"Action   main-action match:{action_main}/{n} = {action_main/n:.1%}  (lenient: subset, mirrors 95.8% rule)")
    print(f"Action   mean Jaccard:     {action_jaccard_sum/n:.3f}")
    print(f"Sensitivity exact match:   {sens_correct}/{sens_total} = {sens_correct/sens_total:.1%}")
    print(f"Externality exact match:   {ext_correct}/{ext_total} = {ext_correct/ext_total:.1%}")
    print(f"\n(Baseline reference — independent LLM vs human GT, their rule:")
    print(f"  Action 95.8% [main-action], Sensitivity 85.7%, Externality 86.6%)")

    print("\nSensitivity confusion (gt -> pred, mismatches only):")
    for (gt, pred), c in sorted(sens_confusion.items(), key=lambda x: -x[1]):
        if gt != pred:
            print(f"  {gt:>24} -> {pred:<24} x{c}")
    print("\nExternality confusion (gt -> pred, mismatches only):")
    for (gt, pred), c in sorted(ext_confusion.items(), key=lambda x: -x[1]):
        if gt != pred:
            print(f"  {gt:>10} -> {pred:<10} x{c}")

    def _group_stats(rows: list[dict]) -> dict:
        if not rows:
            return {}
        na = len(rows)
        se_rows = [r for r in rows if not r["excluded_s_e"]]
        nse = len(se_rows)
        return {
            "n": na,
            "action_main": sum(r["action_main"] for r in rows) / na,
            "action_exact": sum(r["action_exact"] for r in rows) / na,
            "sensitivity": sum(r["pred_sensitivity"] == r["gt_sensitivity"] for r in se_rows) / nse if nse else None,
            "externality": sum(r["pred_externality"] == r["gt_externality"] for r in se_rows) / nse if nse else None,
        }

    dev_split = ns.dev_split
    dev_rows = details[:dev_split]
    test_rows = details[dev_split:]
    dev = _group_stats(dev_rows)
    test = _group_stats(test_rows)

    def _fmt(g: dict) -> str:
        if not g:
            return "(empty)"
        s = f"{g['sensitivity']:.1%}" if g["sensitivity"] is not None else "n/a"
        e = f"{g['externality']:.1%}" if g["externality"] is not None else "n/a"
        return (f"n={g['n']:>3}  Action(main)={g['action_main']:.1%}  "
                f"Action(exact)={g['action_exact']:.1%}  Sens={s}  Ext={e}")

    print("\n" + "=" * 60)
    print("DEV / TEST SPLIT  (dev = prompt-aligned; test = held-out, report this)")
    print("=" * 60)
    print(f"DEV  (items 1..{dev_split}):        {_fmt(dev)}")
    print(f"TEST (items {dev_split+1}..{len(details)}): {_fmt(test)}")
    print(f"IRR reference (human-human):  Action alpha=0.880  Sens kappa=0.886  Ext kappa=0.889")

    out_path = repo_root / ns.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "summary": {
            "n": n,
            "action_main": action_main, "action_main_pct": action_main / n,
            "action_exact": action_exact, "action_exact_pct": action_exact / n,
            "action_mean_jaccard": action_jaccard_sum / n,
            "sensitivity_correct": sens_correct, "sensitivity_total": sens_total,
            "sensitivity_pct": sens_correct / sens_total if sens_total else None,
            "externality_correct": ext_correct, "externality_total": ext_total,
            "externality_pct": ext_correct / ext_total if ext_total else None,
            "server_context": not ns.no_server_context,
            "model": clf._model,
        },
        "dev_group": dev,
        "test_group": test,
        "baseline": meta["agreement_stats"],
        "failures": failures,
        "details": details,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote detailed results to {ns.out} ({len(failures)} failures logged)")


if __name__ == "__main__":
    main()
