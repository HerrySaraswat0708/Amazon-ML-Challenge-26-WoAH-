"""Candidate generation (blocking): GPU TF-IDF nearest-neighbour retrieval.

For every country (an open set of labels, read from the data) and every target
source (S2, S3) separately, three retrievers each return the top-k records:
  name  - char 3-gram TF-IDF cosine on name_norm
  addr  - char 3-gram TF-IDF cosine on addr_norm
  comb  - mean of the two cosines (hstacked unit vectors / sqrt 2)
The candidate set is the union. For every candidate pair we keep the per-retriever
rank and the exact name / address cosines, which later become matcher features.

Usage:
  python -m ber.candidates --split train --queries val --k 20 --gpus 0,1,2,3
  python -m ber.candidates --split test  --queries all --k 20 --gpus 0,1,2,3,4,5,6
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

COLS = ["entity_id", "country", "name_norm", "addr_norm"]
RETRIEVERS = ("name", "addr", "comb")          # forward: S1 -> records
ALL_RETRIEVERS = RETRIEVERS + ("rcomb", "rname")  # + reverse: record -> S1


# --------------------------------------------------------------------------- vectors
def make_vectorizer(max_df: float) -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=max_df,
                           sublinear_tf=True, dtype=np.float32)


def vectorize(q_text: pd.Series, d_text: pd.Series, max_df: float):
    v = make_vectorizer(max_df)
    v.fit(pd.concat([q_text, d_text], ignore_index=True))
    return v.transform(q_text).tocsr(), v.transform(d_text).tocsr()


def combine(a: sparse.csr_matrix, b: sparse.csr_matrix) -> sparse.csr_matrix:
    return (sparse.hstack([a, b], format="csr") * np.float32(1 / np.sqrt(2))).astype(np.float32)


def rowwise_cos(A: sparse.csr_matrix, B: sparse.csr_matrix, ia: np.ndarray, ib: np.ndarray,
                chunk: int = 2_000_000) -> np.ndarray:
    """cos(A[ia[j]], B[ib[j]]) for every j (rows are L2-normalized already)."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        e = s + chunk
        out[s:e] = np.asarray(A[ia[s:e]].multiply(B[ib[s:e]]).sum(axis=1)).ravel()
    return out


# --------------------------------------------------------------------------- GPU top-k
def _to_torch_csr(M: sparse.csr_matrix, device: str):
    import torch
    M = M.tocsr()
    return torch.sparse_csr_tensor(torch.from_numpy(M.indptr.astype(np.int32)),
                                   torch.from_numpy(M.indices.astype(np.int32)),
                                   torch.from_numpy(M.data.astype(np.float32)), size=M.shape, device=device)


def gpu_topk(Q: sparse.csr_matrix, D: sparse.csr_matrix, k: int, device: str,
             chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Top-k columns of Q @ D.T per row.

    Computed as sparse(D) @ dense(Q_chunk.T) on GPU. The query chunk is shipped sparse and
    densified on the device (a CPU toarray + transpose is ~100x slower), and top-k runs along
    contiguous rows of the transposed score block.
    """
    import torch

    k = min(k, D.shape[0])
    Dt = _to_torch_csr(D, device)
    idx_out = np.empty((Q.shape[0], k), dtype=np.int64)
    val_out = np.empty((Q.shape[0], k), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, Q.shape[0], chunk):
            q = _to_torch_csr(Q[s:s + chunk].T.tocsr(), device).to_dense()  # (V, c)
            S = torch.sparse.mm(Dt, q).T.contiguous()                       # (c, nD)
            v, i = S.topk(k, dim=1)
            idx_out[s:s + chunk] = i.cpu().numpy()
            val_out[s:s + chunk] = v.cpu().numpy()
            del S, q
    del Dt
    torch.cuda.empty_cache()
    return idx_out, val_out


# --------------------------------------------------------------------------- one task
def name_text(df: pd.DataFrame) -> pd.Series:
    """Retrieval text for names: name_norm plus the web-domain stem when it adds information."""
    dom = df.name_domain.where(df.name_domain != "", None)
    squashed = df.name_norm.str.replace(" ", "", regex=False)
    add = [(" " + d) if d and d not in s else "" for d, s in zip(dom.fillna(""), squashed)]
    return (df.name_norm + pd.Series(add, index=df.index)).str.strip()


def _long(I: np.ndarray, V: np.ndarray, rows: np.ndarray, col: str, swap: bool = False) -> pd.DataFrame:
    """Top-k matrix -> long (qi, di, rank) frame. rows maps query row -> global row index."""
    k = I.shape[1]
    q = np.repeat(rows, k)
    d = I.ravel()
    keep = V.ravel() > 0
    rank = np.tile(np.arange(1, k + 1, dtype=np.int16), len(rows))
    if swap:  # reverse retrieval: query = S2/S3 record, hit = S1 entity
        q, d = d, q
    return pd.DataFrame({"qi": q[keep], "di": d[keep], col: rank[keep]})


def run_task(task: dict) -> str:
    """task: split, country, src, q_ids_path, k, k_rev, max_df, gpu, out_dir, proc_dir"""
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    t0 = time.time()
    proc = Path(task["proc_dir"])
    cols = COLS + ["name_domain"]
    # the full S1 pool of this country: queries are a subset, but reverse retrieval must
    # see every competing S1 entity to be realistic
    S = pd.read_parquet(proc / f"{task['split']}_source1.parquet", columns=cols)
    S = S[S.country == task["country"]].reset_index(drop=True)
    q_ids = set(pd.read_parquet(task["q_ids_path"]).entity_id)
    q_rows = np.flatnonzero(S.entity_id.isin(q_ids).values)
    src_i = task["src"][1]
    D = pd.read_parquet(proc / f"{task['split']}_source{src_i}.parquet", columns=cols)
    D = D[D.country == task["country"]].reset_index(drop=True)
    tag = f"{task['country']}-{task['src']}"
    if len(q_rows) == 0 or len(D) == 0:
        return f"{tag}: empty"

    Sn, Dn = vectorize(name_text(S), name_text(D), task["max_df"])
    Sa, Da = vectorize(S.addr_norm, D.addr_norm, task["max_df"])
    Sc, Dc = combine(Sn, Sa), combine(Dn, Da)
    t_vec = time.time() - t0

    device = f"cuda:{task['gpu']}"
    parts = []
    # forward: S1 query -> top-k records of this source
    for r, (A, B) in {"name": (Sn, Dn), "addr": (Sa, Da), "comb": (Sc, Dc)}.items():
        I, V = gpu_topk(A[q_rows], B, task["k"], device)
        parts.append(_long(I, V, q_rows, f"rank_{r}"))
    # reverse: every record -> top-k S1 entities (deduplicated pool, fewer look-alikes);
    # name-only reverse for records without an address
    if task["k_rev"]:
        all_d = np.arange(len(D))
        I, V = gpu_topk(Dc, Sc, task["k_rev"], device)
        parts.append(_long(I, V, all_d, "rank_rcomb", swap=True))
        empty = np.flatnonzero((D.addr_norm == "").values)
        if len(empty):
            I, V = gpu_topk(Dn[empty], Sn, task["k_rev"], device)
            parts.append(_long(I, V, empty, "rank_rname", swap=True))
    t_knn = time.time() - t0 - t_vec

    qset = np.zeros(len(S), bool)
    qset[q_rows] = True
    cand = None
    for p in parts:
        p = p[qset[p.qi.values]].groupby(["qi", "di"], as_index=False).min()
        cand = p if cand is None else cand.merge(p, on=["qi", "di"], how="outer")
    for c in cand.columns:
        if c.startswith("rank_"):
            cand[c] = cand[c].fillna(0).astype(np.int16)
    for r in ALL_RETRIEVERS:
        if f"rank_{r}" not in cand:
            cand[f"rank_{r}"] = np.int16(0)
    qi, di = cand.qi.values, cand.di.values
    cand["cos_name"] = rowwise_cos(Sn, Dn, qi, di)
    cand["cos_addr"] = rowwise_cos(Sa, Da, qi, di)
    cand.insert(0, "s1", S.entity_id.values[qi])
    cand.insert(1, "m", D.entity_id.values[di])
    cand = cand.drop(columns=["qi", "di"])

    out = Path(task["out_dir"]) / f"part_{tag}.parquet"
    cand.to_parquet(out, index=False)
    return (f"{tag}: Q={len(q_rows):,} S1pool={len(S):,} D={len(D):,} cands={len(cand):,} "
            f"vec={t_vec:.0f}s knn={t_knn:.0f}s total={time.time() - t0:.0f}s")


def _worker(args):
    task, q = args
    import torch
    torch.cuda.set_device(task["gpu"])
    return run_task(task)


# --------------------------------------------------------------------------- evaluation
def evaluate(cand: pd.DataFrame, pairs: pd.DataFrame, n_queries: int, ks=(5, 10, 20, 30, 50)) -> dict:
    """Pair recall of each retriever / the union at several k, per source and overall."""
    truth = pairs[["s1", "m"]].assign(hit=True)
    c = cand.merge(truth, on=["s1", "m"], how="left")
    c["hit"] = c.hit.fillna(False).astype(bool)
    c["src"] = c.m.str[:2]
    n_true = len(pairs)
    res = {"n_queries": n_queries, "n_true_pairs": n_true, "n_candidates": int(len(c)),
           "cands_per_s1": round(len(c) / n_queries, 1),
           "union_recall": round(c.hit.sum() / n_true * 100, 3)}
    kmax = int(c[[f"rank_{r}" for r in ALL_RETRIEVERS]].max().max())
    for k in [k for k in ks if k <= kmax]:
        row = {}
        for r in ALL_RETRIEVERS:
            sel = (c[f"rank_{r}"] > 0) & (c[f"rank_{r}"] <= k)
            row[r] = round(c.hit[sel].sum() / n_true * 100, 3)
        sel = np.zeros(len(c), bool)
        for r in ALL_RETRIEVERS:
            sel |= (c[f"rank_{r}"] > 0) & (c[f"rank_{r}"] <= k)
        row["union"] = round(c.hit[sel].sum() / n_true * 100, 3)
        row["cands_per_s1"] = round(sel.sum() / n_queries, 1)
        res[f"k={k}"] = row
    by = {}
    tsrc = pairs.m.str[:2]
    for s in ("S2", "S3"):
        by[s] = round(c.hit[c.src == s].sum() / max(1, (tsrc == s).sum()) * 100, 3)
    res["union_recall_by_src"] = by
    return res


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-dir", default="data/processed")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--queries", default="val", help="val | train | all  (train split only: val/train)")
    ap.add_argument("--sample", type=int, default=0, help="subsample this many query entities")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--k-rev", type=int, default=5, help="reverse retrieval top-k (0 = off)")
    ap.add_argument("--max-df", type=float, default=0.2)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    proc = Path(a.proc_dir)
    name = a.out or f"data/candidates/{a.split}_{a.queries}{'_s' + str(a.sample) if a.sample else ''}_k{a.k}"
    out_dir = Path(name)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1 = pd.read_parquet(proc / f"{a.split}_source1.parquet", columns=["entity_id", "country"])
    if a.split == "train" and a.queries in ("val", "train"):
        sp = pd.read_parquet(proc / "split.parquet")
        s1 = s1[s1.entity_id.isin(set(sp.source1_entity_id[sp.fold == a.queries]))]
    if a.sample:
        s1 = s1.sample(min(a.sample, len(s1)), random_state=0)
    q_path = out_dir / "queries.parquet"
    s1[["entity_id", "country"]].to_parquet(q_path, index=False)
    countries = sorted(s1.country.unique())
    print(f"queries: {len(s1):,}  countries: {countries}")

    gpus = [int(g) for g in a.gpus.split(",")]
    tasks = []
    for c in countries:
        for src in ("S2", "S3"):
            tasks.append({"split": a.split, "country": c, "src": src, "q_ids_path": str(q_path),
                          "k": a.k, "k_rev": a.k_rev, "max_df": a.max_df, "proc_dir": str(proc), "out_dir": str(out_dir)})
    # biggest tasks first, round-robin over GPUs
    size = s1.country.value_counts()
    tasks.sort(key=lambda t: -size[t["country"]])
    for i, t in enumerate(tasks):
        t["gpu"] = gpus[i % len(gpus)]

    t0 = time.time()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(len(gpus), len(tasks))) as pool:
        for msg in pool.imap_unordered(_worker, [(t, None) for t in tasks]):
            print(f"[{time.time() - t0:.0f}s] {msg}", flush=True)

    cand = pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("part_*.parquet"))], ignore_index=True)
    cand.to_parquet(out_dir / "candidates.parquet", index=False)
    print(f"candidates: {len(cand):,} ({len(cand) / len(s1):.1f} per S1)  [{time.time() - t0:.0f}s]")

    if a.split == "train":
        pairs = pd.read_parquet(proc / "train_pairs.parquet")
        pairs = pairs[pairs.s1.isin(set(s1.entity_id))]
        rep = evaluate(cand, pairs, len(s1))
        rep["by_country"] = {}
        cty = s1.set_index("entity_id").country
        for c in countries:
            ids = set(cty.index[cty == c])
            rep["by_country"][c] = evaluate(cand[cand.s1.isin(ids)], pairs[pairs.s1.isin(ids)], len(ids))
        with open(out_dir / "recall.json", "w") as f:
            json.dump(rep, f, indent=1)
        print(json.dumps({k: v for k, v in rep.items() if k != "by_country"}, indent=1))
        for c, r in rep["by_country"].items():
            print(c, "union", r["union_recall"], "by_src", r["union_recall_by_src"], "cands/S1", r["cands_per_s1"])


if __name__ == "__main__":
    main()
