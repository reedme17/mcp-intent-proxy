"""Score the non-LLM baselines against ground truth (§5.2 comparison).

Usage:
    uv run python -m eval.run_baselines --gt /path/to/ground-truth-120.json

Action-only, same criteria as the LLM eval (exact-set + main-action subset
match), so the numbers sit directly beside the classifier's. No API calls.
"""

from __future__ import annotations

import argparse

from eval.dataset import load_eval_items
from eval.baselines import keyword_classify, oauth_scope_classify, embedding_nn_classify


def _norm(actions: list[str]) -> set[str]:
    # Normalize the GT's inconsistent SPEND label (SPEND/TRANSACT -> SPEND).
    return {"SPEND" if a == "SPEND/TRANSACT" else a for a in actions}


def _main_match(pred: set[str], gt: set[str]) -> bool:
    if not pred or not gt:
        return pred == gt
    if not (pred & gt):
        return False
    return pred <= gt or gt <= pred


def _score(name: str, preds: list[list[str]], gts: list[list[str]]) -> None:
    n = len(gts)
    exact = main = 0
    for p, g in zip(preds, gts):
        ps, gs = _norm(p), _norm(g)
        exact += int(ps == gs)
        main += int(_main_match(ps, gs))
    print(f"{name:16} exact-set {exact:>3}/{n} = {exact/n:5.1%}   "
          f"main-action {main:>3}/{n} = {main/n:5.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score non-LLM action baselines")
    parser.add_argument("--gt", required=True)
    parser.add_argument("--corpus-dir", default=None)
    ns = parser.parse_args()

    items, meta = load_eval_items(ns.gt, ns.corpus_dir)
    print(f"Loaded {meta['n_joined']}/{meta['n_entries']} items. Action-only, no API.\n")

    gts = [it.gt_action for it in items]
    texts = [f"{it.tool_name}. {it.description}" for it in items]

    kw = [keyword_classify(it.tool_name, it.description) for it in items]
    oa = [oauth_scope_classify(it.tool_name, it.description) for it in items]
    nn = embedding_nn_classify(texts, gts)

    print("ACTION baselines vs ground truth (n=%d):" % len(items))
    print("-" * 64)
    _score("keyword", kw, gts)
    _score("oauth-scope", oa, gts)
    _score("embedding-NN", nn, gts)
    print("-" * 64)
    print("LLM classifier (reference, full 120): "
          "exact 72.5%   main-action 90.0%")
    print("Baseline reference (initial LLM, main rule): action 95.8%")


if __name__ == "__main__":
    main()
