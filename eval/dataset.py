"""Load and join evaluation data from an external ground-truth corpus.

The corpus lives outside this repository (paths are passed in at runtime).
Nothing from it is written back here. Two files are consumed:

- ground-truth-120.json: adjudicated labels (gt_action/sensitivity/externality)
  keyed by a "server :: tool" identifier.
- labeled-dataset-part*.md: verbatim tool descriptions + input schemas, keyed
  by the same identifier.

They are joined on the identifier so the classifier sees exactly the metadata
the human annotators saw.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field


@dataclass
class EvalItem:
    tool_id: str
    server: str
    tool_name: str  # bare tool name without the [param=value] branch suffix
    param_constraint: str  # e.g. "action=add"; empty for whole-tool entries
    description: str
    schema_text: str
    gt_action: list[str]
    gt_sensitivity: str
    gt_externality: str
    excluded_s_e: bool


def _split_branch(tool_id: str) -> tuple[str, str]:
    """Split 'server :: name [action=add]' identifiers.

    Some ground-truth entries are parameter-level: one physical tool split by
    the value of a switching parameter (e.g. Spotify's action=get vs add),
    each branch adjudicated separately because its capability differs. The
    branch constraint lives only in the '[param=value]' suffix — the verbatim
    description is shared across branches — so we extract it as an explicit
    constraint rather than leaving it buried in the identifier string.

    Returns (bare_tool_name, param_constraint). param_constraint is "" when
    the entry is a whole-tool entry.
    """
    # tool_id is the part after '::' already, or the full "server :: tool".
    tail = tool_id.split("::")[-1].strip()
    m = re.match(r"^(.*?)\s*\[(.+?)\]\s*$", tail)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return tail, ""


def _parse_labeled_datasets(corpus_dir: str) -> dict[str, dict]:
    """Parse labeled-dataset-part*.md into {tool_id: {description, schema_text}}."""
    tools: dict[str, dict] = {}
    pattern = os.path.join(corpus_dir, "labeled-dataset-part*.md")
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        server = None
        i = 0
        while i < len(lines):
            line = lines[i]
            m_srv = re.match(r"^## SERVER:\s*(.+?)\s*$", line)
            if m_srv:
                server = m_srv.group(1).strip()
            m_tool = re.match(r"^###\s+(.+?)\s*$", line)
            if m_tool:
                tool_id = m_tool.group(1).strip()
                description = ""
                schema_lines: list[str] = []
                in_schema = False
                j = i + 1
                while j < len(lines) and not lines[j].startswith("###") and not lines[j].startswith("## "):
                    body = lines[j]
                    m_desc = re.match(r"^-\s+\*\*Description\*\*.*?:\s*(.*)$", body)
                    if m_desc:
                        description = m_desc.group(1).strip()
                        in_schema = False
                    elif re.match(r"^-\s+\*\*Input schema\*\*", body):
                        in_schema = True
                    elif re.match(r"^-\s+\*\*", body):
                        in_schema = False
                    elif in_schema and body.strip():
                        schema_lines.append(body.rstrip())
                    j += 1
                tools[tool_id] = {
                    "server": server,
                    "description": description,
                    "schema_text": "\n".join(schema_lines).strip(),
                }
            i += 1
    return tools


def _merge_map(taxonomy: dict, dimension: str) -> dict[str, str]:
    """Build {raw_value: merged_value} from the taxonomy's *_merges block."""
    merges = taxonomy.get(f"{dimension}_merges", {})
    out: dict[str, str] = {}
    for merged, raws in merges.items():
        for raw in raws:
            out[raw] = merged
    return out


def load_eval_items(gt_path: str, corpus_dir: str | None = None) -> tuple[list[EvalItem], dict]:
    """Join ground truth with tool schemas. Returns (items, gt_metadata).

    Sensitivity ground-truth values are normalized through the taxonomy's
    merge map so they are comparable to the classifier's merged output
    (e.g. content-dependent -> non-specific).
    """
    corpus_dir = corpus_dir or os.path.dirname(gt_path)
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    taxonomy = gt.get("taxonomy", {})
    sens_merge = _merge_map(taxonomy, "sensitivity")

    schemas = _parse_labeled_datasets(corpus_dir)

    items: list[EvalItem] = []
    missing: list[str] = []
    for entry in gt["entries"]:
        tool_id = entry["tool"]
        meta = schemas.get(tool_id)
        if meta is None:
            missing.append(tool_id)
            continue
        gt_sens_raw = entry["gt_sensitivity"]
        gt_sens = sens_merge.get(gt_sens_raw, gt_sens_raw)
        bare_name, constraint = _split_branch(tool_id)
        items.append(
            EvalItem(
                tool_id=tool_id,
                server=meta["server"] or "",
                tool_name=bare_name,
                param_constraint=constraint,
                description=meta["description"],
                schema_text=meta["schema_text"],
                gt_action=entry["gt_action"],
                gt_sensitivity=gt_sens,
                gt_externality=entry["gt_externality"],
                excluded_s_e=bool(entry.get("excluded_S_E", False)),
            )
        )

    meta = {
        "n_entries": len(gt["entries"]),
        "n_joined": len(items),
        "n_missing": len(missing),
        "missing": missing,
        "agreement_stats": gt.get("agreement_stats", {}),
    }
    return items, meta
