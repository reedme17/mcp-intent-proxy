"""Single-deny generalization correctness (§5.6), computed offline.

"Deny once, cover everywhere" executes deterministically: after the user denies
a category X, a tool is blocked iff the CLASSIFIER labels it X. So the
correctness of generalization is fully determined by classification accuracy —
no new model runs needed. We simulate denying each action category in turn and,
against the expert GT, count:

- over-block: tool blocked (pred contains X) but GT does not contain X
  (a legitimate tool wrongly caught — annoying, not dangerous).
- under-block: tool should be blocked (GT contains X) but pred does not
  (a tool that should have been stopped slips through — the dangerous
  direction, asymmetric cost).

Reads the already-computed full-120 classifier results; no API calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ACTIONS = [
    "READ/SEARCH", "CREATE", "MODIFY/MANAGE", "DELETE", "SEND",
    "SPEND", "EXECUTE", "PHYSICAL", "CREDENTIAL/IDENTITY", "OTHER",
]


def _norm(labels: list[str]) -> set[str]:
    # GT uses SPEND/TRANSACT in one entry; fold into SPEND.
    return {"SPEND" if a == "SPEND/TRANSACT" else a for a in labels}


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-deny generalization correctness")
    parser.add_argument("--results", default="eval/full120_results.json",
                        help="classifier results with per-tool gt_action + pred_action")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="fail-closed confidence threshold (default 0.6)")
    parser.add_argument("--out", default="eval/generalization_results.json")
    ns = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    details = json.loads((repo_root / ns.results).read_text())["details"]

    # low_conf tools trigger fail-closed: they are escalated to ask/deny
    # regardless of pred, i.e. they are always "blocked" (never silently
    # forwarded). ask counts as "blocked" here because the user gets a chance
    # to veto — the security-relevant question is silent pass-through.
    tools = [{"gt": _norm(d["gt_action"]), "pred": _norm(d["pred_action"]),
              "low_conf": d.get("confidence", 1.0) < ns.threshold} for d in details]
    n = len(tools)
    n_low = sum(t["low_conf"] for t in tools)

    print(f"Single-deny generalization over {n} tools, per action category.")
    print(f"fail-closed threshold={ns.threshold} ({n_low}/{n} tools are low-confidence).\n")
    print(f"{'category':20} {'should':>6} | {'under-block (raw / +fail-closed)':>34} | {'over-block (raw / +fail-closed)':>34}")
    print("-" * 100)

    rows = []
    for x in ACTIONS:
        should = [t for t in tools if x in t["gt"]]
        should_not = [t for t in tools if x not in t["gt"]]

        # RAW: block iff pred contains X.
        under_raw = [t for t in should if x not in t["pred"]]
        over_raw = [t for t in should_not if x in t["pred"]]

        # +FAIL-CLOSED: low-conf tools are always blocked (escalated).
        #  under: should-block but NOT blocked = high-conf AND pred lacks X.
        under_fc = [t for t in should if (not t["low_conf"]) and x not in t["pred"]]
        #  over: should-not-block but blocked = low-conf (escalated) OR (high-conf AND pred has X).
        over_fc = [t for t in should_not if t["low_conf"] or x in t["pred"]]

        def rate(k, tot):
            return len(k) / len(tot) if tot else None
        small = len(should) < 10
        rows.append({
            "category": x, "should_block_n": len(should), "should_not_n": len(should_not),
            "under_raw_n": len(under_raw), "under_raw_rate": rate(under_raw, should),
            "under_fc_n": len(under_fc), "under_fc_rate": rate(under_fc, should),
            "over_raw_n": len(over_raw), "over_raw_rate": rate(over_raw, should_not),
            "over_fc_n": len(over_fc), "over_fc_rate": rate(over_fc, should_not),
            "small_sample": small,
        })

        def fmt(k, tot):
            return f"{len(k)/len(tot):.0%} ({len(k)}/{len(tot)})" if tot else "n/a"
        ub = f"{fmt(under_raw,should):>15} / {fmt(under_fc,should):>15}"
        ob = f"{fmt(over_raw,should_not):>15} / {fmt(over_fc,should_not):>15}"
        flag = "  n<10" if small else ""
        print(f"{x:20} {len(should):>6} | {ub} | {ob}{flag}")

    big = [r for r in rows if not r["small_sample"]]
    ts = sum(r["should_block_n"] for r in big)
    tsn = sum(r["should_not_n"] for r in big)
    tur, tuf = sum(r["under_raw_n"] for r in big), sum(r["under_fc_n"] for r in big)
    tor_, tof = sum(r["over_raw_n"] for r in big), sum(r["over_fc_n"] for r in big)
    print("-" * 100)
    print(f"Aggregate (n>=10 categories):")
    print(f"  under-block: raw {tur}/{ts}={tur/ts:.0%}  ->  +fail-closed {tuf}/{ts}={tuf/ts:.0%}  (should-block missed, dangerous)")
    print(f"  over-block:  raw {tor_}/{tsn}={tor_/tsn:.0%}  ->  +fail-closed {tof}/{tsn}={tof/tsn:.0%}  (legit blocked, annoying)")
    print("(fail-closed lowers dangerous under-block, raises annoying over-block — the intended trade.)")

    (repo_root / ns.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nWrote {ns.out}")


if __name__ == "__main__":
    main()
