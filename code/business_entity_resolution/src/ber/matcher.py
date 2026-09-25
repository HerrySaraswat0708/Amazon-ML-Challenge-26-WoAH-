"""Two-pass pairwise matcher + assignment + threshold tuning.

Pass 1: LightGBM on retrieval + pairwise + retrieval-context features.
Pass 2: LightGBM on pass-1 features + context features computed from pass-1 scores
        (collective ER: each pair is judged against its competitors). Train-fold
        pass-1 scores are out-of-fold so pass 2 never sees leaked labels.
Decision: each S2/S3 record goes to at most its best-scoring S1 entity, kept if
          p >= threshold (tuned on validation macro F0.5, singletons included).

Usage:
  python -m ber.matcher build  --split train --cands data/candidates/train_all/candidates.parquet
  python -m ber.matcher train  --train-s1 600000
  python -m ber.matcher predict --split test --cands data/candidates/test_all/candidates.parquet
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import context_features, load_records, pairwise_features, token_idf
from .io import f05_entity

FEAT_DIR = Path("data/features")
MODEL_DIR = Path("models")
_G: dict = {}


# --------------------------------------------------------------------------- feature building
def _feat_chunk(bounds):
    a, b = bounds
    return pairwise_features(_G["cand"].iloc[a:b], _G["s1"], _G["oth"], _G["idf"])


def build_features(split: str, cands_path: str, proc: Path, workers: int, chunk: int = 500_000) -> Path:
    t0 = time.time()
    cand = pd.read_parquet(cands_path)
    s1, oth = load_records(proc, split)
    idf = token_idf(pd.concat([s1.name_core, oth.name_core, s1.addr_norm, oth.addr_norm]))
    print(f"[{time.time()-t0:.0f}s] loaded {len(cand):,} candidates, idf vocab {len(idf):,}", flush=True)
    _G.update(cand=cand, s1=s1, oth=oth, idf=idf)
    bounds = [(i, min(i + chunk, len(cand))) for i in range(0, len(cand), chunk)]
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_feat_chunk, bounds)
    F = pd.concat(parts)
    print(f"[{time.time()-t0:.0f}s] pairwise features {F.shape}", flush=True)
    F = pd.concat([cand[["s1", "m"]], F, context_features(cand, F.cos_mean, "ret")], axis=1)
    if split == "train":
        pairs = pd.read_parquet(proc / "train_pairs.parquet")
        truth = pd.MultiIndex.from_frame(pairs[["s1", "m"]])
        F["label"] = pd.MultiIndex.from_frame(F[["s1", "m"]]).isin(truth).astype(np.int8)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out = FEAT_DIR / f"{split}_pass1.parquet"
    F.to_parquet(out, index=False)
    print(f"[{time.time()-t0:.0f}s] wrote {out} {F.shape}", flush=True)
    return out


# --------------------------------------------------------------------------- model helpers
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, num_threads=64, verbose=-1)


def feat_cols(F: pd.DataFrame) -> list[str]:
    return [c for c in F.columns if c not in ("s1", "m", "label", "fold")]


def fit(F: pd.DataFrame, cols: list[str], rounds: int, valid: pd.DataFrame | None = None) -> lgb.Booster:
    dtr = lgb.Dataset(F[cols], F.label, free_raw_data=True)
    vs = [lgb.Dataset(valid[cols], valid.label, reference=dtr)] if valid is not None else []
    cb = [lgb.log_evaluation(100)] + ([lgb.early_stopping(50)] if vs else [])
    return lgb.train(PARAMS, dtr, num_boost_round=rounds, valid_sets=vs, callbacks=cb)


# --------------------------------------------------------------------------- decision + metric
def assign(F: pd.DataFrame, p: np.ndarray, threshold: float) -> dict[str, set[str]]:
    """Each record -> its best S1 (ties: first), kept if p >= threshold."""
    d = pd.DataFrame({"s1": F.s1.values, "m": F.m.values, "p": p})
    d = d[d.p >= threshold]
    d = d.sort_values("p", ascending=False).drop_duplicates("m")
    return d.groupby("s1").m.apply(set).to_dict()


def macro_f05(pred: dict, truth: dict, s1_ids) -> float:
    return float(np.mean([f05_entity(pred.get(s, set()), truth.get(s, set())) for s in s1_ids]))


def tune_threshold(F: pd.DataFrame, p: np.ndarray, truth: dict, s1_ids, grid=None) -> tuple[float, dict]:
    grid = grid if grid is not None else np.round(np.arange(0.20, 0.96, 0.025), 3)
    scores = {float(t): macro_f05(assign(F, p, t), truth, s1_ids) for t in grid}
    best = max(scores, key=scores.get)
    return best, scores


# --------------------------------------------------------------------------- training
def train(proc: Path, train_s1: int, rounds: int, seed: int = 0) -> None:
    t0 = time.time()
    F = pd.read_parquet(FEAT_DIR / "train_pass1.parquet")
    split = pd.read_parquet(proc / "split.parquet").set_index("source1_entity_id").fold
    F["fold"] = split.reindex(F.s1.values).values
    cols1 = feat_cols(F)
    print(f"[{time.time()-t0:.0f}s] features {F.shape}, positives {F.label.mean():.3%}", flush=True)

    # --- pass 1 with 2-fold out-of-fold scores on the train fold
    rng = np.random.default_rng(seed)
    tr_ids = F.s1[F.fold == "train"].unique()
    half = pd.Series(rng.random(len(tr_ids)) < 0.5, index=tr_ids)
    F["half"] = -1
    is_tr = (F.fold == "train").values
    F.loc[is_tr, "half"] = half.reindex(F.s1[is_tr].values).values.astype(int)
    val = F[F.fold == "val"]
    p1 = np.zeros(len(F), np.float32)
    for h in (0, 1):
        ids = tr_ids[half.values == h]
        ids = ids[rng.permutation(len(ids))[: train_s1 // 2]]
        trn = F[F.s1.isin(set(ids))]
        es = val.sample(min(len(val), 2_000_000), random_state=h)
        m = fit(trn, cols1, rounds, es)
        other = (F.half == 1 - h).values
        p1[other] = m.predict(F.loc[other, cols1], num_threads=64)
        p1[(F.fold == "val").values] += 0.5 * m.predict(val[cols1], num_threads=64)
        m.save_model(str(MODEL_DIR / f"pass1_h{h}.txt"))
        print(f"[{time.time()-t0:.0f}s] pass1 half {h}: trained on {len(trn):,} rows, {m.best_iteration} rounds", flush=True)
    F["p1"] = p1

    # --- pass 2: context from pass-1 scores
    ctx = context_features(F[["s1", "m"]], F.p1, "p1")
    F = pd.concat([F, ctx], axis=1)
    cols2 = cols1 + ["p1"] + list(ctx.columns)
    trn_ids = rng.permutation(tr_ids)[:train_s1]
    trn = F[F.s1.isin(set(trn_ids))]
    val = F[F.fold == "val"]
    m2 = fit(trn, cols2, rounds, val.sample(min(len(val), 2_000_000), random_state=7))
    m2.save_model(str(MODEL_DIR / "pass2.txt"))
    p2 = m2.predict(val[cols2], num_threads=64)
    print(f"[{time.time()-t0:.0f}s] pass2 trained, {m2.best_iteration} rounds", flush=True)

    # --- evaluate on the validation fold (all val S1 entities, singletons included)
    pairs = pd.read_parquet(proc / "train_pairs.parquet")
    val_ids = split.index[split == "val"]
    vp = pairs[pairs.s1.isin(set(val_ids))]
    truth = vp.groupby("s1").m.apply(set).to_dict()
    report = {"cols_pass1": cols1, "cols_pass2": cols2}
    for name, p in (("pass1", val.p1.values), ("pass2", p2)):
        t, curve = tune_threshold(val, p, truth, val_ids)
        pred = assign(val, p, t)
        f_by = {}
        cty = pd.read_parquet(proc / "train_source1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
        for c in sorted(cty.reindex(val_ids).unique()):
            ids = [s for s in val_ids if cty[s] == c]
            f_by[c] = round(macro_f05(pred, truth, ids), 5)
        tp = sum(len(pred.get(s, set()) & truth.get(s, set())) for s in val_ids)
        npred = sum(len(v) for v in pred.values())
        report[name] = {"threshold": t, "macro_f05": round(curve[t], 5), "by_country": f_by,
                        "pair_precision": round(tp / max(1, npred), 5), "pair_recall": round(tp / len(vp), 5),
                        "curve": {str(k): round(v, 5) for k, v in curve.items()}}
        print(f"{name}: threshold={t} macro F0.5={curve[t]:.5f} by country {f_by} "
              f"pairP={tp/max(1,npred):.4f} pairR={tp/len(vp):.4f}", flush=True)
    imp = pd.Series(m2.feature_importance("gain"), index=cols2).sort_values(ascending=False)
    report["importance_pass2"] = (imp / imp.sum()).round(4).head(30).to_dict()
    json.dump(report, open(MODEL_DIR / "train_report.json", "w"), indent=1)
    val.assign(p2=p2)[["s1", "m", "label", "p1", "p2"]].to_parquet(FEAT_DIR / "val_scores.parquet", index=False)
    print(f"[{time.time()-t0:.0f}s] done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "train"])
    ap.add_argument("--proc-dir", default="data/processed")
    ap.add_argument("--split", default="train")
    ap.add_argument("--cands")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--train-s1", type=int, default=600_000)
    ap.add_argument("--rounds", type=int, default=1500)
    a = ap.parse_args()
    MODEL_DIR.mkdir(exist_ok=True)
    if a.cmd == "build":
        build_features(a.split, a.cands, Path(a.proc_dir), a.workers)
    else:
        train(Path(a.proc_dir), a.train_s1, a.rounds)


if __name__ == "__main__":
    main()
