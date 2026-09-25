"""Preprocess all sources: validation split, fit resources, normalize every record.

Usage: python -m ber.preprocess --raw-dir data/raw --out-dir data/processed [--workers 32]

Outputs (data/processed/):
  split.parquet                    source1_entity_id, fold ("train" | "val")
  train_pairs.parquet              one row per labelled (S1, S2|S3) match, with fold
  resources.json                   learned native-script tables (fit on train fold only)
  {train,test}_source{1,2,3}.parquet  normalized records
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .normalize import Resources, normalize_address, normalize_name
from .resources import fit_resources, load_resources, save_resources

VAL_FRAC = 0.10
SEED = 42

_RES: Resources | None = None


def _init(res: Resources) -> None:
    global _RES
    _RES = res


def _norm_chunk(df: pd.DataFrame) -> pd.DataFrame:
    names = pd.DataFrame([normalize_name(x, _RES) for x in df.business_name], index=df.index)
    addrs = pd.DataFrame([normalize_address(x, _RES) for x in df.business_address], index=df.index)
    return pd.concat([df, names, addrs], axis=1)


def normalize_frame(df: pd.DataFrame, res: Resources, workers: int) -> pd.DataFrame:
    bounds = np.linspace(0, len(df), max(1, workers * 4) + 1).astype(int)
    chunks = [df.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    with Pool(workers, initializer=_init, initargs=(res,)) as pool:
        parts = pool.map(_norm_chunk, chunks)
    return pd.concat(parts)


def build_pairs(raw: Path) -> pd.DataFrame:
    gt = pd.read_parquet(raw / "train_ground_truth.parquet")
    ids = gt.matched_entity_ids.str.split(",")
    p = pd.DataFrame({"s1": gt.source1_entity_id.repeat(ids.str.len()).values,
                      "m": ids.explode().values})
    return p[p.m.notna() & (p.m != "")].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--reuse-resources", action="store_true")
    a = ap.parse_args()
    raw, out = Path(a.raw_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    s1 = pd.read_parquet(raw / "train_source1.parquet")
    rng = np.random.default_rng(SEED)
    fold = np.where(rng.random(len(s1)) < VAL_FRAC, "val", "train")
    split = pd.DataFrame({"source1_entity_id": s1.entity_id, "fold": fold})
    split.to_parquet(out / "split.parquet", index=False)
    print(f"split: {pd.Series(fold).value_counts().to_dict()}")

    pairs = build_pairs(raw).merge(split, left_on="s1", right_on="source1_entity_id").drop(columns="source1_entity_id")
    pairs.to_parquet(out / "train_pairs.parquet", index=False)

    res_path = out / "resources.json"
    if a.reuse_resources and res_path.exists():
        res = load_resources(res_path)
    else:
        recs = pd.concat([pd.read_parquet(raw / f"train_source{i}.parquet") for i in (1, 2, 3)]).set_index("entity_id")
        tp = pairs[pairs.fold == "train"]
        pr = pd.DataFrame({
            "s1_name": recs.business_name.reindex(tp.s1).values,
            "s1_addr": recs.business_address.reindex(tp.s1).values,
            "m_name": recs.business_name.reindex(tp.m).values,
            "m_addr": recs.business_address.reindex(tp.m).values,
        })
        del recs
        res = fit_resources(pr)
        save_resources(res, res_path)
    print(f"resources: native_state={len(res.native_state)} native_token={len(res.native_token)} "
          f"({time.time() - t0:.0f}s)")

    for split_name in ("train", "test"):
        for i in (1, 2, 3):
            df = pd.read_parquet(raw / f"{split_name}_source{i}.parquet")
            nd = normalize_frame(df, res, a.workers)
            nd.to_parquet(out / f"{split_name}_source{i}.parquet", index=False)
            print(f"{split_name}_source{i}: {len(nd):,} rows ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
