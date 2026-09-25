"""Data loading, submission writing and local F0.5 scoring."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str | Path) -> pd.DataFrame:
    # keep_default_na=False so empty ID lists / "NA" names stay strings
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def load_split(data_dir: str | Path, split: str) -> dict[str, pd.DataFrame]:
    """Load {s1, s2, s3[, gt]} for split in {"train", "test"}."""
    d = Path(data_dir) / split
    out = {f"s{i}": read_tsv(d / f"{split}_source{i}.tsv") for i in (1, 2, 3)}
    gt = d / f"{split}_ground_truth.tsv"
    if gt.exists():
        out["gt"] = read_tsv(gt)
    return out


def parse_id_list(s: str) -> list[str]:
    return [x for x in s.split(",") if x] if s else []


def gt_to_dict(gt: pd.DataFrame) -> dict[str, set[str]]:
    return {r.source1_entity_id: set(parse_id_list(r.matched_entity_ids)) for r in gt.itertuples()}


def write_id_lists(mapping: dict[str, list[str]], s1_ids: list[str], path: str | Path, col: str) -> None:
    """Write one row per S1 id (in s1_ids order), deduplicating each list."""
    rows = [(sid, ",".join(dict.fromkeys(mapping.get(sid, [])))) for sid in s1_ids]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for sid, ids in rows:
            f.write(f"{sid}\t{ids}\n")


def write_matching(mapping, s1_ids, path):
    write_id_lists(mapping, s1_ids, path, "matched_entity_ids")


def write_candidates(mapping, s1_ids, path):
    write_id_lists(mapping, s1_ids, path, "candidate_entity_ids")


def f05_entity(pred: set[str], true: set[str], beta: float = 0.5) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f05(pred: dict[str, set[str]], true: dict[str, set[str]]) -> float:
    """Macro-averaged F0.5 over all S1 ids in `true` (singletons included)."""
    return sum(f05_entity(set(pred.get(k, ())), v) for k, v in true.items()) / len(true)


def candidate_recall(cands: dict[str, set[str]], true: dict[str, set[str]]) -> float:
    """Pair-level recall ceiling of a blocking stage."""
    tot = sum(len(v) for v in true.values())
    hit = sum(len(set(cands.get(k, ())) & v) for k, v in true.items())
    return hit / tot if tot else 1.0
