"""Convert raw TSVs to parquet (string dtype, no NA coercion) for fast reloads.

Usage: python -m ber.convert --data-dir dataset --out-dir data/raw
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def read_tsv_fast(path: Path) -> pd.DataFrame:
    # QUOTE_NONE: fields are never quoted in this data; stray '"' must stay literal.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="data/raw")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for f in sorted((Path(a.data_dir) / split).glob("*.tsv")):
            df = read_tsv_fast(f)
            with open(f, encoding="utf-8") as fh:
                n_lines = sum(1 for _ in fh) - 1
            print(f"{f.name}: rows={len(df):,} file_lines={n_lines:,} cols={list(df.columns)}")
            df.to_parquet(out / f"{f.stem}.parquet", index=False)


if __name__ == "__main__":
    main()
