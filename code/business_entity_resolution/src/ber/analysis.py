"""Post-cleaning EDA: pair similarity, name collisions, distractors, blocking-key recall.

Usage: python -m ber.analysis --proc-dir data/processed --out reports/eda_stats.json
Writes a JSON of all statistics plus a few CSV tables next to it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

SAMPLE_PAIRS = 300_000
SEED = 0
MAX_BLOCK = 2000  # blocks larger than this are skipped (reported)


def load(proc: Path, split: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(proc / f"{split}_source{i}.parquet") for i in (1, 2, 3)]).set_index("entity_id")


def tok_jaccard(a: str, b: str) -> float:
    A, B = set(a.split()), set(b.split())
    return len(A & B) / len(A | B) if A and B else 0.0


def pair_features(L: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    f = {}
    f["name_raw_tsr"] = [fuzz.token_set_ratio(a.lower(), b.lower()) for a, b in zip(L.business_name, R.business_name)]
    f["name_norm_tsr"] = [fuzz.token_set_ratio(a, b) for a, b in zip(L.name_norm, R.name_norm)]
    f["name_core_tsr"] = [fuzz.token_set_ratio(a, b) for a, b in zip(L.name_core, R.name_core)]
    f["name_core_jacc"] = [tok_jaccard(a, b) for a, b in zip(L.name_core, R.name_core)]
    f["name_core_exact"] = (L.name_core.values == R.name_core.values)
    f["addr_raw_tsr"] = [fuzz.token_set_ratio(a.lower(), b.lower()) for a, b in zip(L.business_address, R.business_address)]
    f["addr_norm_tsr"] = [fuzz.token_set_ratio(a, b) for a, b in zip(L.addr_norm, R.addr_norm)]
    f["addr_empty"] = (R.addr_norm.values == "")
    f["house_eq"] = (L.addr_house.values == R.addr_house.values) & (R.addr_house.values != "")
    f["state_eq"] = (L.addr_state.values == R.addr_state.values) & (R.addr_state.values != "")
    f["state_missing"] = (R.addr_state.values == "")
    f["is_translit"] = R.name_is_translit.values
    f["has_alias"] = (R.name_alias.values != "")
    f["legal_eq"] = (L.name_legal.values == R.name_legal.values)
    return pd.DataFrame(f)


def describe(df: pd.DataFrame, group: pd.Series | None = None) -> dict:
    cols = [c for c in df.columns if df[c].dtype != bool]
    bcols = [c for c in df.columns if df[c].dtype == bool]
    def one(d):
        r = {c: {"mean": round(float(d[c].mean()), 2), "p10": float(np.percentile(d[c], 10)),
                 "p50": float(np.percentile(d[c], 50))} for c in cols}
        r.update({c: round(float(d[c].mean()) * 100, 2) for c in bcols})
        r["n"] = int(len(d))
        return r
    if group is None:
        return one(df)
    return {str(k): one(g) for k, g in df.groupby(group.values)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-dir", default="data/processed")
    ap.add_argument("--out", default="reports/eda_stats.json")
    a = ap.parse_args()
    proc, out = Path(a.proc_dir), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats: dict = {}

    tr = load(proc, "train")
    pairs = pd.read_parquet(proc / "train_pairs.parquet")
    split = pd.read_parquet(proc / "split.parquet")
    s1 = tr[tr.index.str.startswith("S1-")]
    oth = tr[~tr.index.str.startswith("S1-")]
    matched = set(pairs.m)

    # ---------------- positive-pair similarity (raw vs cleaned), by source and country
    sp = pairs.sample(min(SAMPLE_PAIRS, len(pairs)), random_state=SEED)
    L, R = tr.loc[sp.s1], tr.loc[sp.m]
    pf = pair_features(L, R)
    pf["src"] = sp.m.str[:2].values
    pf["country"] = L.country.values
    grp = pf.src + "-" + pf.country
    stats["pos_pairs_by_src_country"] = describe(pf.drop(columns=["src", "country"]), grp)
    stats["pos_pairs_all"] = describe(pf.drop(columns=["src", "country"]))

    # hard-case taxonomy of positives
    lowname = pf.name_core_tsr < 50
    lowaddr = (pf.addr_norm_tsr < 50) | pf.addr_empty
    stats["pos_taxonomy_pct"] = {
        "name_strong_addr_strong": round(float((~lowname & ~lowaddr).mean() * 100), 2),
        "name_weak_addr_strong (address-only match)": round(float((lowname & ~lowaddr).mean() * 100), 2),
        "name_strong_addr_weak_or_empty (name-only match)": round(float((~lowname & lowaddr).mean() * 100), 2),
        "both_weak": round(float((lowname & lowaddr).mean() * 100), 2),
    }
    ex = pd.DataFrame({"s1_name": L.business_name.values, "s1_addr": L.business_address.values,
                       "m_name": R.business_name.values, "m_addr": R.business_address.values,
                       "name_core_tsr": pf.name_core_tsr, "addr_norm_tsr": pf.addr_norm_tsr})
    ex[lowname & lowaddr].head(40).to_csv(out.parent / "pos_both_weak_examples.csv", index=False)
    ex[lowname & ~lowaddr].head(40).to_csv(out.parent / "pos_address_only_examples.csv", index=False)

    # ---------------- random negatives within same country+state (easy baseline) and name-collision negatives
    rng = np.random.default_rng(SEED)
    neg_r = oth.sample(len(sp), random_state=SEED, replace=True)
    same_c = L.country.values == neg_r.country.values
    nf = pair_features(L[same_c], neg_r[same_c])
    stats["random_neg_same_country"] = describe(nf)

    # ---------------- name ambiguity: identical core names across distinct S1 entities / S1 vs other clusters
    vc = s1.name_core.value_counts()
    stats["s1_name_core_dup"] = {
        "s1_entities": int(len(s1)),
        "distinct_name_core": int(len(vc)),
        "pct_s1_sharing_name_core_with_another_s1": round(float((s1.name_core.map(vc) > 1).mean() * 100), 2),
        "top_shared_names": vc.head(15).to_dict(),
    }
    # hard negatives: S2/S3 records whose name_core exactly equals an S1 name_core but belong elsewhere
    s1_core_to_ids = s1.reset_index().groupby("name_core").entity_id.apply(list)
    pos_set = set(zip(pairs.s1, pairs.m))
    samp = oth.sample(200_000, random_state=SEED)
    hits = samp.name_core.map(s1_core_to_ids).dropna()
    n_pos = n_neg = 0
    for mid, ids in hits.items():
        for sid in ids:
            if (sid, mid) in pos_set:
                n_pos += 1
            else:
                n_neg += 1
    stats["exact_name_core_blocking_precision"] = {
        "pairs": n_pos + n_neg, "true": n_pos, "false": n_neg,
        "precision_pct": round(100 * n_pos / max(1, n_pos + n_neg), 2)}

    # ---------------- distractors: unmatched S2/S3 records
    oth_matched = oth.index.isin(list(matched))
    stats["distractors"] = {
        "S2_unmatched_pct": round(float((~oth_matched[oth.index.str.startswith("S2")]).mean() * 100), 2),
        "S3_unmatched_pct": round(float((~oth_matched[oth.index.str.startswith("S3")]).mean() * 100), 2),
        "unmatched_name_core_in_s1_pct": round(float(oth[~oth_matched].name_core.isin(set(s1.name_core)).mean() * 100), 2),
        "matched_name_core_in_s1_pct": round(float(oth[oth_matched].name_core.isin(set(s1.name_core)).mean() * 100), 2),
        "unmatched_addr_empty_pct": round(float((oth[~oth_matched].addr_norm == "").mean() * 100), 2),
        "matched_addr_empty_pct": round(float((oth[oth_matched].addr_norm == "").mean() * 100), 2),
        "unmatched_translit_pct": round(float(oth[~oth_matched].name_is_translit.mean() * 100), 2),
        "matched_translit_pct": round(float(oth[oth_matched].name_is_translit.mean() * 100), 2),
    }
    oth[~oth_matched].sample(40, random_state=SEED)[["business_name", "business_address", "country"]] \
        .to_csv(out.parent / "distractor_examples.csv")

    # ---------------- cluster structure vs S1 features
    gt = pairs.groupby("s1").size()
    s1n = s1.assign(n_match=gt.reindex(s1.index).fillna(0).astype(int).values)
    stats["cluster_size_by_country"] = s1n.groupby("country").n_match.describe().round(3).to_dict(orient="index")
    s1n["name_ntok"] = s1n.name_core.str.split().str.len()
    stats["singleton_rate_by_name_ntok"] = s1n.groupby(s1n.name_ntok.clip(upper=7)).n_match.apply(lambda x: round(float((x == 0).mean() * 100), 2)).to_dict()
    stats["singleton_rate_by_legal"] = s1n.groupby(s1n.name_legal.replace("", "<none>")).n_match.agg(
        lambda x: round(float((x == 0).mean() * 100), 2)).sort_values().tail(12).to_dict()

    # ---------------- blocking-key recall on the validation fold (pair recall + candidate volume)
    val_ids = set(split.source1_entity_id[split.fold == "val"])
    vpairs = pairs[pairs.s1.isin(val_ids)]
    s1v = s1[s1.index.isin(val_ids)]
    first_tok = lambda s: s.str.split().str[0].fillna("")
    keys = {
        "name_core_exact": (lambda d: d.country + "|" + d.name_core),
        "name_first_tok+state": (lambda d: d.country + "|" + first_tok(d.name_core) + "|" + d.addr_state),
        "name_first_tok": (lambda d: d.country + "|" + first_tok(d.name_core)),
        "house+first_street_tok": (lambda d: d.country + "|" + d.addr_house + "|" +
                                   d.addr_norm.str.split().str[1].fillna("")),
    }
    block = {}
    cand_sets = {}
    for k, fn in keys.items():
        kl = fn(s1v).rename("key").reset_index()
        kr = fn(oth).rename("key").reset_index()
        kl = kl[~kl.key.str.endswith("|")]
        kr = kr[~kr.key.str.endswith("|")]
        sizes = kr.key.value_counts()
        big = set(sizes.index[sizes > MAX_BLOCK])
        s1_in_big = kl.key.isin(big).mean()
        cand = kl[~kl.key.isin(big)].merge(kr[~kr.key.isin(big)], on="key", suffixes=("_s1", "_m"))
        cs = set(zip(cand.entity_id_s1, cand.entity_id_m))
        cand_sets[k] = cs
        hit = sum((s, m) in cs for s, m in zip(vpairs.s1, vpairs.m))
        block[k] = {"pair_recall_pct": round(100 * hit / len(vpairs), 2), "candidates": len(cs),
                    "cands_per_s1": round(len(cs) / len(s1v), 1), "max_block_raw": int(sizes.max()),
                    "pct_s1_in_dropped_oversize_blocks": round(float(s1_in_big) * 100, 2)}
    union = set().union(*cand_sets.values())
    hit = sum((s, m) in union for s, m in zip(vpairs.s1, vpairs.m))
    block["UNION"] = {"pair_recall_pct": round(100 * hit / len(vpairs), 2), "candidates": len(union),
                      "cands_per_s1": round(len(union) / len(s1v), 1)}
    stats["blocking_key_recall_val"] = block

    # ---------------- test-set shift: France vs others on cleaned fields
    te = load(proc, "test")
    stats["test_country_profile"] = te.assign(src=te.index.str[:2]).groupby(["src", "country"]).agg(
        n=("name_norm", "size"),
        addr_empty_pct=("addr_norm", lambda x: round(float((x == "").mean() * 100), 2)),
        state_found_pct=("addr_state", lambda x: round(float((x != "").mean() * 100), 2)),
        legal_found_pct=("name_legal", lambda x: round(float((x != "").mean() * 100), 2)),
        alias_pct=("name_alias", lambda x: round(float((x != "").mean() * 100), 2)),
    ).reset_index().to_dict(orient="records")
    stats["train_country_profile"] = tr.assign(src=tr.index.str[:2]).groupby(["src", "country"]).agg(
        n=("name_norm", "size"),
        addr_empty_pct=("addr_norm", lambda x: round(float((x == "").mean() * 100), 2)),
        state_found_pct=("addr_state", lambda x: round(float((x != "").mean() * 100), 2)),
        legal_found_pct=("name_legal", lambda x: round(float((x != "").mean() * 100), 2)),
        alias_pct=("name_alias", lambda x: round(float((x != "").mean() * 100), 2)),
    ).reset_index().to_dict(orient="records")
    fr = te[te.country == "France"]
    stats["france_top_legal"] = fr.name_legal.replace("", "<none>").value_counts().head(12).to_dict()
    stats["france_top_core_tokens"] = fr.name_core.str.split().explode().value_counts().head(30).to_dict()

    with open(out, "w") as fh:
        json.dump(stats, fh, indent=1, default=str)
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
