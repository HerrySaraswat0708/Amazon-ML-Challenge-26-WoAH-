"""Pairwise features for (S1 entity, S2/S3 record) candidate pairs.

Three groups:
  retrieval  - ranks from each retriever and TF-IDF cosines (from ber.candidates)
  pairwise   - string / number / field comparisons of the two records
  context    - how this pair compares with the competing pairs around it
               (other candidates of the same S1, other S1s claiming the same record)
No feature uses the country label, so unseen countries (France) are handled the same way.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz, process

REC_COLS = ["entity_id", "name_norm", "name_core", "name_legal", "name_alias", "name_domain",
            "name_is_translit", "addr_norm", "addr_state", "addr_house", "addr_numbers", "addr_postcode"]
RANK_COLS = ["rank_name", "rank_addr", "rank_comb", "rank_rcomb", "rank_rname"]
NO_RANK = 99
CP_WORKERS = -1  # rapidfuzz threads; set to 1 inside multiprocessing workers


def load_records(proc, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    s1 = pd.read_parquet(proc / f"{split}_source1.parquet", columns=REC_COLS).set_index("entity_id")
    oth = pd.concat([pd.read_parquet(proc / f"{split}_source{i}.parquet", columns=REC_COLS) for i in (2, 3)]
                    ).set_index("entity_id")
    return s1, oth


def token_idf(texts: pd.Series) -> dict[str, float]:
    df = Counter()
    for t in texts:
        df.update(set(t.split()))
    n = len(texts)
    return {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}


def _cp(a, b, scorer) -> np.ndarray:
    return process.cpdist(list(a), list(b), scorer=scorer, workers=CP_WORKERS).astype(np.float32)


def idf_overlap(a: pd.Series, b: pd.Series, idf: dict, default: float) -> tuple[np.ndarray, np.ndarray]:
    """IDF-weighted share of A's tokens found in B, and weighted Jaccard."""
    cov = np.zeros(len(a), np.float32)
    jac = np.zeros(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        A, B = set(x.split()), set(y.split())
        if not A or not B:
            continue
        w = lambda s: sum(idf.get(t, default) for t in s)
        inter = w(A & B)
        cov[i] = inter / w(A)
        jac[i] = inter / w(A | B)
    return cov, jac


def num_jaccard(a: pd.Series, b: pd.Series) -> np.ndarray:
    out = np.full(len(a), -1, np.float32)  # -1 = one side has no numbers
    for i, (x, y) in enumerate(zip(a, b)):
        A, B = set(x.split()), set(y.split())
        if A and B:
            out[i] = len(A & B) / len(A | B)
    return out


def house_match(a: pd.Series, b: pd.Series) -> np.ndarray:
    """2 exact, 1 one is a suffix/prefix of the other (dropped digit: 301 vs 4301), 0 differ, -1 missing."""
    out = np.full(len(a), -1, np.int8)
    for i, (x, y) in enumerate(zip(a, b)):
        if x and y:
            out[i] = 2 if x == y else (1 if (x.endswith(y) or y.endswith(x) or x.startswith(y) or y.startswith(x)) else 0)
    return out


def field_eq(a: pd.Series, b: pd.Series) -> np.ndarray:
    """1 equal, 0 different, -1 either missing."""
    a, b = a.values, b.values
    out = np.where((a == "") | (b == ""), -1, np.where(a == b, 1, 0))
    return out.astype(np.int8)


def pairwise_features(cand: pd.DataFrame, s1: pd.DataFrame, oth: pd.DataFrame, idf: dict) -> pd.DataFrame:
    L = s1.loc[cand.s1.values]
    R = oth.loc[cand.m.values]
    f = pd.DataFrame(index=cand.index)
    f["src_s3"] = (cand.m.str[:2] == "S3").astype(np.int8).values
    for c in RANK_COLS:
        f[c] = cand[c].replace(0, NO_RANK).astype(np.int16).values if c in cand else np.int16(NO_RANK)
    f["n_retrievers"] = (f[RANK_COLS] < NO_RANK).sum(axis=1).astype(np.int8).values
    f["cos_name"] = cand.cos_name.values
    f["cos_addr"] = cand.cos_addr.values
    f["cos_mean"] = (f.cos_name + f.cos_addr) / 2

    # names
    ln, rn = L.name_core.values, R.name_core.values
    f["nm_ratio"] = _cp(ln, rn, fuzz.ratio)
    f["nm_tset"] = _cp(ln, rn, fuzz.token_set_ratio)
    f["nm_tsort"] = _cp(ln, rn, fuzz.token_sort_ratio)
    f["nm_partial"] = _cp(ln, rn, fuzz.partial_ratio)
    f["nm_jw"] = _cp(ln, rn, distance.JaroWinkler.normalized_similarity)
    f["nm_full_tset"] = _cp(L.name_norm.values, R.name_norm.values, fuzz.token_set_ratio)
    default = max(idf.values())
    f["nm_idf_cov_l"], f["nm_idf_jac"] = idf_overlap(L.name_core, R.name_core, idf, default)
    f["nm_idf_cov_r"], _ = idf_overlap(R.name_core, L.name_core, idf, default)
    f["nm_exact"] = (ln == rn).astype(np.int8)
    f["nm_first_eq"] = (L.name_core.str.split().str[0].values == R.name_core.str.split().str[0].values).astype(np.int8)
    f["nm_ntok_l"] = L.name_core.str.count(" ").add(1).values.astype(np.int8)
    f["nm_ntok_diff"] = (R.name_core.str.count(" ").values - L.name_core.str.count(" ").values).astype(np.int8)
    f["legal_eq"] = field_eq(L.name_legal, R.name_legal)
    # squashed S1 name vs record's domain / alias (catches 'reditprivate.com', 'Jaxaria formerly X')
    l_sq = L.name_norm.str.replace(" ", "", regex=False).values
    f["dom_ratio"] = np.where(R.name_domain.values != "", _cp(l_sq, R.name_domain.values, fuzz.partial_ratio), -1)
    f["alias_ratio"] = np.where(R.name_alias.values != "", _cp(ln, R.name_alias.values, fuzz.token_set_ratio), -1)
    f["r_translit"] = R.name_is_translit.values.astype(np.int8)
    f["r_name_empty"] = (rn == "").astype(np.int8)

    # addresses
    la, ra = L.addr_norm.values, R.addr_norm.values
    f["r_addr_empty"] = (ra == "").astype(np.int8)
    f["ad_tset"] = np.where(ra != "", _cp(la, ra, fuzz.token_set_ratio), -1)
    f["ad_tsort"] = np.where(ra != "", _cp(la, ra, fuzz.token_sort_ratio), -1)
    f["ad_partial"] = np.where(ra != "", _cp(la, ra, fuzz.partial_ratio), -1)
    f["ad_idf_cov_r"], _ = idf_overlap(R.addr_norm, L.addr_norm, idf, default)
    f["ad_ntok_ratio"] = (R.addr_norm.str.count(" ").values + 1) / (L.addr_norm.str.count(" ").values + 1)
    f["num_jacc"] = num_jaccard(L.addr_numbers, R.addr_numbers)
    f["house"] = house_match(L.addr_house, R.addr_house)
    f["state_eq"] = field_eq(L.addr_state, R.addr_state)
    f["postcode_eq"] = field_eq(L.addr_postcode, R.addr_postcode)
    for c in f.columns[f.dtypes == np.float64]:
        f[c] = f[c].astype(np.float32)
    return f


def _group_rank(key: pd.Series, score: pd.Series) -> np.ndarray:
    return score.groupby(key.values).rank(ascending=False, method="min").values.astype(np.float32)


def _second_best(key: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Second-highest score within each key group, broadcast back (-1 if the group has one row)."""
    order = np.lexsort((-score, key))
    k_sorted, s_sorted = key[order], score[order]
    first = np.r_[True, k_sorted[1:] != k_sorted[:-1]]
    is_second = np.r_[False, first[:-1]] & ~first
    second_val = pd.Series(s_sorted[is_second], index=k_sorted[is_second])
    return pd.Series(key).map(second_val).fillna(-1.0).values.astype(np.float32)


def context_features(cand: pd.DataFrame, score: pd.Series, prefix: str) -> pd.DataFrame:
    """Competition features around each pair, from any per-pair score (retrieval or model).

    per S1 entity : rank of this record, gap to the best record, # records near the top
    per record    : rank of this S1, gap to the best competing S1, # S1s claiming it
    """
    s = pd.Series(score.values, index=cand.index)
    g1, gm = cand.s1.values, cand.m.values
    out = pd.DataFrame(index=cand.index)
    best_s1 = s.groupby(g1).transform("max")
    best_m = s.groupby(gm).transform("max")
    out[f"{prefix}_rank_in_s1"] = _group_rank(cand.s1, s)
    out[f"{prefix}_gap_s1_best"] = (best_s1 - s).values
    out[f"{prefix}_rank_in_m"] = _group_rank(cand.m, s)
    out[f"{prefix}_gap_m_best"] = (best_m - s).values
    # margin over the runner-up competing S1 for this record (positive = this S1 wins)
    second_m = _second_best(gm, s.values)
    out[f"{prefix}_margin_m"] = np.where(s.values >= best_m.values, s.values - second_m, s.values - best_m.values)
    out[f"{prefix}_n_s1_for_m"] = s.groupby(gm).transform("size").values.astype(np.int16)
    return out
