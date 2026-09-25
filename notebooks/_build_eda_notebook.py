"""Generates notebooks/01_eda_preprocessing_blocking.ipynb. Run: python notebooks/_build_eda_notebook.py"""
from pathlib import Path

import nbformat as nbf

cells: list[tuple[str, str]] = []
md = lambda s: cells.append(("md", s.strip("\n")))
code = lambda s: cells.append(("code", s.strip("\n")))

# =============================================================================
md(r"""
# Business Entity Resolution — Preprocessing, Visualization, Analysis & Blocking

This notebook is the interactive version of `reports/EDA_REPORT.md`. It covers:

1. **Setup & loading** — raw vs. normalized records
2. **Ground-truth structure** — cluster sizes, singletons, the one-S1-per-record constraint
3. **Noise catalogue** — what's dirty, where, and how often
4. **Preprocessing playground** — run the normalizer on any string and see every step
5. **Did cleaning help?** — similarity of true matches vs. non-matches, before/after
6. **Where the difficulty is** — name collisions, distractors, hard positives
7. **Blocking, explained** — what it is, why it's needed, worked examples, recall vs. cost
8. **A better blocker (demo)** — TF-IDF nearest-neighbour search on one state
9. **France** — the unseen test country

**Prerequisite** (once, ~12 min): from the repo root
```bash
export PYTHONPATH=code/business_entity_resolution/src
python -m ber.convert && python -m ber.preprocess --workers 32
```
Kernel: **Python (amazon-er)**.
""")

code(r"""
import sys, json, re, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from rapidfuzz import fuzz

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from ber.normalize import normalize_name, normalize_address, squash_phonetic
from ber.resources import load_resources

RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"
RES = load_resources(PROC / "resources.json")

pd.set_option("display.max_colwidth", 90); pd.set_option("display.width", 200)

# ---- chart style: one recessive grid, thin marks, fixed categorical order ----
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
C = {"US": BLUE, "India": ORANGE, "France": AQUA}
plt.rcParams.update({
    "figure.dpi": 110, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c9c8c2", "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "font.size": 9.5, "legend.frameon": False,
})
def bar_labels(ax, fmt="{:,.0f}", **kw):
    for c in ax.containers: ax.bar_label(c, fmt=fmt, padding=2, fontsize=8, color="#52514e", **kw)
""")

code(r"""
# Load normalized train + test. ~12M rows each; ~1-2 min and ~15 GB RAM.
# Set SAMPLE_FRAC < 1 to load a random fraction of S2/S3 if memory is tight
# (cluster-level stats then become partial).
SAMPLE_FRAC = 1.0
COLS = ["entity_id", "business_name", "business_address", "country", "name_norm", "name_core",
        "name_legal", "name_alias", "name_domain", "name_is_translit", "name_script",
        "addr_norm", "addr_state", "addr_house", "addr_numbers"]

def load(split):
    parts = []
    for i in (1, 2, 3):
        d = pd.read_parquet(PROC / f"{split}_source{i}.parquet", columns=COLS)
        if i > 1 and SAMPLE_FRAC < 1: d = d.sample(frac=SAMPLE_FRAC, random_state=0)
        parts.append(d.assign(src=f"S{i}"))
    return pd.concat(parts, ignore_index=True).set_index("entity_id")

t0 = time.time()
tr, te = load("train"), load("test")
pairs = pd.read_parquet(PROC / "train_pairs.parquet")      # one row per true (S1, S2|S3) match
split = pd.read_parquet(PROC / "split.parquet")            # 90/10 S1 split for validation
print(f"train {len(tr):,}  test {len(te):,}  true pairs {len(pairs):,}  ({time.time()-t0:.0f}s)")
tr.sample(5, random_state=3)
""")

# =============================================================================
md(r"""
## 1. What the data looks like

Every file has the same 4 columns: `entity_id` (prefix = source), `business_name`, `business_address`, `country`.
The ground truth lists, for each Source-1 entity, the S2/S3 records that are the same business.
Below: one real cluster. S1 is the clean reference; S2/S3 are noisy copies.
""")

code(r"""
def show_cluster(s1_id, df=tr):
    ids = [s1_id] + pairs.loc[pairs.s1 == s1_id, "m"].tolist()
    return df.loc[ids, ["src", "business_name", "business_address", "name_norm", "addr_norm", "addr_state"]]

show_cluster("S1-706225498")   # Cabrera Secure Sciences
""")

code(r"""
# Re-run this cell for random clusters (change the country too)
rid = pairs.s1[tr.loc[pairs.s1, "country"].values == "India"].sample(1).iloc[0]
show_cluster(rid)
""")

code(r"""
# Record counts per source and country
cnt = pd.concat([tr.assign(split="train"), te.assign(split="test")]) \
        .groupby(["split", "src", "country"]).size().unstack("country").fillna(0).astype(int)
ax = cnt[["US", "India", "France"]].plot.bar(color=[C["US"], C["India"], C["France"]], width=0.8,
                                             edgecolor="#fcfcfb", linewidth=2, figsize=(9, 3.4))
ax.set_title("Records per source and country"); ax.set_xlabel(""); ax.set_ylabel("records")
ax.yaxis.set_major_formatter(lambda v, _: f"{v/1e6:.1f}M"); ax.tick_params(axis="x", rotation=0)
ax.legend(title=None, ncols=3, loc="upper left"); plt.show()
cnt
""")

# =============================================================================
md(r"""
## 2. Ground-truth structure

Things to notice:
- The typical S1 entity has **3–4 matches**; only **~5.6% are singletons** (no match).
- S2 and S3 each contain **several copies** of the same business (up to 5 and 6).
- **No S2/S3 record is matched to more than one S1** → a record can be assigned to at most one entity.
- **Matches never cross countries.**
""")

code(r"""
n_match = pairs.groupby("s1").size().reindex(split.source1_entity_id).fillna(0).astype(int)
per_src = pairs.assign(src=pairs.m.str[:2]).groupby(["s1", "src"]).size().unstack(fill_value=0) \
               .reindex(split.source1_entity_id).fillna(0).astype(int)

fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
vc = n_match.value_counts().sort_index()
axs[0].bar(vc.index, vc.values, color=[GRAY if i == 0 else BLUE for i in vc.index], width=0.8)
axs[0].set_title("Matches per S1 entity (gray = singletons)"); axs[0].set_xlabel("# matched S2+S3 records")
axs[0].yaxis.set_major_formatter(lambda v, _: f"{v/1e3:.0f}k"); axs[0].set_xticks(vc.index)
w = 0.4
for j, (s, col) in enumerate([("S2", BLUE), ("S3", ORANGE)]):
    v = per_src[s].value_counts().sort_index()
    axs[1].bar(v.index + (j - 0.5) * w, v.values, width=w, color=col, label=s, edgecolor="#fcfcfb")
axs[1].set_title("Copies of the same business within one source"); axs[1].set_xlabel("# matches from that source")
axs[1].yaxis.set_major_formatter(lambda v, _: f"{v/1e3:.0f}k"); axs[1].legend()
plt.tight_layout(); plt.show()

m_counts = pairs.m.value_counts()
print(f"singletons: {(n_match==0).mean():.2%}   mean matches: {n_match.mean():.2f}")
print(f"S2/S3 records matched to >1 S1: {(m_counts>1).sum()}")
cc = tr.loc[pairs.s1, "country"].values != tr.loc[pairs.m, "country"].values
print(f"cross-country true pairs: {cc.sum()}")
for s in ("S2", "S3"):
    ids = tr.index[tr.src == s]
    print(f"{s} records with no S1 match (distractors): {1 - ids.isin(pairs.m).mean():.1%}")
""")

md(r"""
**Why this matters for the metric.** F0.5 is computed *per S1 entity*, then averaged. For one entity:

$F_{0.5} = \dfrac{1.25\,P\,R}{0.25\,P + R}$

A singleton scores 1 if we predict nothing, 0 otherwise. Let's feel the precision/recall asymmetry:
""")

code(r"""
from ber.io import f05_entity
truth = {"a", "b", "c", "d"}   # an entity with 4 true matches
for name, pred in [("perfect", truth), ("miss 2 (P=1, R=.5)", {"a", "b"}), ("1 extra wrong (P=.8, R=1)", truth | {"x"}),
                   ("2 extra wrong", truth | {"x", "y"}), ("predict nothing", set()), ("only 1 right", {"a"})]:
    print(f"{name:28s} F0.5 = {f05_entity(pred, truth):.3f}")
print("singleton, predict nothing  ->", f05_entity(set(), set()))
print("singleton, predict 1 wrong  ->", f05_entity({"x"}, set()))
""")

# =============================================================================
md(r"""
## 3. Noise catalogue

Rates (% of records) of each noise pattern, per split/source/country, measured on the **raw** text.
S1 is clean; S2 shouts in ALL-CAPS addresses; S3 writes US states in full and has alias markers ("X formerly Y").
""")

code(r"""
prof = pd.DataFrame(json.load(open(RAW / "_noise_profile.json")))
prof["group"] = prof.split + " " + prof.src + " " + prof.country
cols = ["addr_empty", "name_all_upper", "addr_all_upper", "name_nonascii_latin", "name_junk_prefix", "name_brackets",
        "name_domain", "name_alias", "name_digit_in_word", "name_double_space", "addr_has_nonlatin",
        "addr_near", "addr_door", "addr_unit", "addr_lead_zero", "addr_cdp", "addr_us_zip5"]
prof.set_index("group")[cols].T.style.background_gradient(cmap="Blues", axis=None, vmin=0, vmax=30).format("{:.1f}")
""")

code(r"""
# Script of the business name (India, S2 vs S3). Non-Latin names are phonetic transliterations.
ind = tr[(tr.country == "India") & tr.src.isin(["S2", "S3"])]
sc = ind.groupby("src").name_script.value_counts(normalize=True).mul(100).unstack(0).drop("LATIN").sort_values("S2")
ax = sc.plot.barh(color=[BLUE, ORANGE], width=0.8, figsize=(7, 3.6), edgecolor="#fcfcfb")
ax.set_title("Non-Latin scripts in India business names (% of records)"); ax.set_xlabel("%"); ax.set_ylabel("")
plt.show()
ind[ind.name_is_translit][["business_name", "name_norm"]].sample(8, random_state=1)
""")

# =============================================================================
md(r"""
## 4. Preprocessing playground

`normalize_name` / `normalize_address` (in `src/ber/normalize.py`) produce these fields:

| field | meaning |
|---|---|
| `name_norm` | lowercased, accent-free, punctuation-free, OCR-fixed, alias-resolved, transliterated |
| `name_core` | `name_norm` minus legal forms and stopwords — the "identity" part |
| `name_legal` | canonical legal forms found (`inc`, `llc`, `pvt ltd`, `sarl`…) |
| `name_alias` / `name_domain` | the other half of "X dba Y", or the stem of a web domain |
| `addr_norm` | state removed, abbreviations canonicalized, null parts dropped, leading zeros stripped |
| `addr_state` | canonical state/region code (works for codes, full names, native scripts) |
| `addr_house` / `addr_numbers` | numeric tokens used as matching keys |

**Edit the strings below and re-run.**
""")

code(r"""
names = ["Cabrera 5ecure Sciences LP", "Jaxaria Formerly Cabrera Secure Sciences", "emerakeystonecom",
         "Elite  + C0mpany", "Lawrence Venmfes Private (Limited)", "एसएस फूड प्राइवेट लिमिटेड",
         "ગોલ્ડ મીડિયા પ્રાઇવેટ લિમિટેડ", "Maison Àzar SASU", "-- Holloway Peak Inc Seafood"]
pd.DataFrame([{"raw": n, **normalize_name(n, RES)} for n in names]).drop(columns=["name_script"])
""")

code(r"""
addrs = ["00709 Hackberry Saint, Tilden, Texas", "1344 MCDONALD HILL ROAD, CHILLICTHE CDP, OH",
         "165 Barren River Drive, # UNIT 2, Erlanger Ky, Kentucky", "DL, North West, No C-956 Jd-36b, New Delhi",
         "Rajkot, Aqua, Flat B- 1002, New 150 Feet Ring Road, ગુજરાત", "N°23 R. D'arras, Lille, Nord", "NULL"]
pd.DataFrame([{"raw": a, **normalize_address(a, RES)} for a in addrs])
""")

md(r"""
**How transliteration works.** Indic names are word-by-word *phonetic* spellings of English words
(`प्राइवेट` = "private"). We romanize with `unidecode` (→ `praaivett`), reduce to a consonant skeleton
(→ `prvt`), and look that up in a dictionary **learned from training pairs** (aligning native tokens with
the S1 name). Unknown words fall back to the romanization.
""")

code(r"""
from unidecode import unidecode
for w in ["प्राइवेट", "लिमिटेड", "मार्केटिंग", "லிமிடெட்", "ലോജിസ്റ്റിക്സ്"]:
    rom = unidecode(w).lower(); sk = squash_phonetic(rom)
    print(f"{w:14s} unidecode={rom:18s} skeleton={sk:8s} -> {RES.native_token.get(sk, '(not in dictionary)')}")
print(f"\nlearned: {len(RES.native_token)} tokens, {len(RES.native_state)} native state names")
print(RES.native_state)
""")

# =============================================================================
md(r"""
## 5. Did cleaning help?

For a sample of **true pairs** and **random non-pairs from the same country**, compare the
token-set similarity (0–100) of raw vs. cleaned fields. We want the blue (true) mass pushed right and
separated from the orange (random) mass.
""")

code(r"""
N = 50_000
sp = pairs.sample(N, random_state=0)
L, R = tr.loc[sp.s1], tr.loc[sp.m]
oth = tr[tr.src != "S1"]
negR = oth.sample(N, random_state=1)
keep = negR.country.values == L.country.values          # random partner from the same country
negL, negR = L[keep], negR[keep]

def sims(A, B):
    return pd.DataFrame({
        "name raw":    [fuzz.token_set_ratio(a.lower(), b.lower()) for a, b in zip(A.business_name, B.business_name)],
        "name clean":  [fuzz.token_set_ratio(a, b) for a, b in zip(A.name_core, B.name_core)],
        "addr raw":    [fuzz.token_set_ratio(a.lower(), b.lower()) for a, b in zip(A.business_address, B.business_address)],
        "addr clean":  [fuzz.token_set_ratio(a, b) for a, b in zip(A.addr_norm, B.addr_norm)],
    })
pos, neg = sims(L, R), sims(negL, negR)

fig, axs = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
bins = np.linspace(0, 100, 41)
for ax, c in zip(axs, pos.columns):
    ax.hist(neg[c], bins, color=ORANGE, alpha=.75, label="random same-country pair")
    ax.hist(pos[c], bins, color=BLUE, alpha=.75, label="true match")
    ax.set_title(c); ax.set_xlabel("token-set ratio")
axs[0].set_ylabel("pairs"); axs[0].legend(loc="upper left", fontsize=8)
plt.tight_layout(); plt.show()
pd.DataFrame({"true: mean": pos.mean(), "true: p10": pos.quantile(.1), "random: mean": neg.mean()}).round(1)
""")

code(r"""
# Where did cleaning help most? True-pair name similarity by source x country, raw vs clean
g = pd.concat([pos, pd.DataFrame({"grp": sp.m.str[:2].values + " " + L.country.values})], axis=1)
ax = g.groupby("grp")[["name raw", "name clean"]].mean().plot.bar(color=[GRAY, BLUE], width=.75, figsize=(7, 3), edgecolor="#fcfcfb")
ax.set_ylim(60, 100); ax.set_title("Mean name similarity of true pairs"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=0)
bar_labels(ax, "{:.1f}"); plt.show()
""")

# =============================================================================
md(r"""
## 6. Where the difficulty is

Random non-pairs are easy (similarity ≈ 33). The hard cases are:

**(a) Name collisions.** Many *different* businesses share the same name. Half of all S1 entities share
their exact core name with another S1 entity. So the address has to break the tie.
""")

code(r"""
s1 = tr[tr.src == "S1"]
vc = s1.name_core.value_counts()
print(f"S1 entities sharing core name with another S1: {(s1.name_core.map(vc) > 1).mean():.1%}")
ax = vc.head(15)[::-1].plot.barh(color=BLUE, figsize=(6, 3.8), width=.75)
ax.set_title("Most repeated S1 core names (distinct businesses!)"); ax.set_xlabel("# S1 entities"); plt.show()
s1[s1.name_core == "meridian"][["business_name", "business_address"]].head(8)
""")

md(r"""
**(b) Distractors.** ~26% of S2/S3 records belong to *no* S1 entity, and many carry realistic
names that also appear in S1. The model must be willing to say "none of these".

**(c) Hard positives.** A few % of true matches have a completely different name (invented brand,
squashed domain) *or* an empty address. Use the slider values below to explore.
""")

code(r"""
NAME_T, ADDR_T = 50, 50
lowN, lowA = pos["name clean"] < NAME_T, (pos["addr clean"] < ADDR_T) | (R.addr_norm.values == "")
tax = pd.Series({"both strong": (~lowN & ~lowA).mean(), "address-only (name weak)": (lowN & ~lowA).mean(),
                 "name-only (address weak/empty)": (~lowN & lowA).mean(), "both weak": (lowN & lowA).mean()}) * 100
print(tax.round(2).to_string())
ex = pd.DataFrame({"S1 name": L.business_name.values, "S1 addr": L.business_address.values,
                   "match name": R.business_name.values, "match addr": R.business_address.values})
print("\naddress-only examples:"); display(ex[(lowN & ~lowA).values].head(6))
print("name-only examples:");      display(ex[(~lowN & lowA).values].head(6))
""")

# =============================================================================
md(r"""
## 7. Blocking, explained

### The problem
To find matches for one S1 entity we'd ideally compare it with **every** S2/S3 record. Let's count:
""")

code(r"""
n_s1_test = (te.src == "S1").sum(); n_oth_test = (te.src != "S1").sum()
print(f"test: {n_s1_test:,} S1 x {n_oth_test:,} S2/S3 = {n_s1_test*n_oth_test:.2e} comparisons")
print(f"at 1 µs per comparison (optimistic for an ML model): {n_s1_test*n_oth_test/1e6/86400:,.0f} days")
print(f"true matches per S1 ≈ 3.5  ->  fraction of comparisons that matter: {3.5/n_oth_test:.1e}")
""")

md(r"""
That's impossible, and 99.99997% of those comparisons are obviously non-matches ("Cabrera Secure Sciences,
Kentucky" vs. "Ram Care Pvt Ltd, Bangalore").

### The idea
**Blocking = a cheap first stage that throws away obvious non-matches** so that the expensive matching
model only looks at a short list of plausible candidates per S1 entity.

```
10M S2/S3 records ──[ blocking: cheap, high-recall ]──> ~20-50 candidates per S1 ──[ ML matcher: expensive, precise ]──> final matches
```

The classic way: compute a **blocking key** for every record; only records with the **same key** are compared.
Records sharing a key form a **block**.

| key | idea | "Cabrera Secure Sciences, 165 Barren River Dr, Erlanger, KY" → key |
|---|---|---|
| exact core name | same normalized name | `US\|cabrera secure sciences` |
| first name token + state | same first word, same state | `US\|cabrera\|ky` |
| house no. + street word | same building | `US\|165\|barren` |

A blocking stage is judged by two numbers:
- **Pair recall** (a.k.a. *pair completeness*) = % of true matches that survive into the candidate set.
  **This caps your final recall** — a match dropped here can never be recovered.
- **Candidates per S1** (or *reduction ratio*) = how much work is left for the model. Fewer = faster, and
  also *easier* for the model (fewer look-alikes to confuse it).

Multiple keys are usually **unioned**: each catches matches the others miss.

### Worked example
""")

code(r"""
first_tok = lambda s: s.str.split().str[0].fillna("")
KEYS = {
    "exact core name":          lambda d: d.country + "|" + d.name_core,
    "first name token + state": lambda d: d.country + "|" + first_tok(d.name_core) + "|" + d.addr_state,
    "house no. + street word":  lambda d: d.country + "|" + d.addr_house + "|" + d.addr_norm.str.split().str[1].fillna(""),
}
t0 = time.time()
oth = tr[tr.src != "S1"]
oth_keys = pd.DataFrame({k: f(oth) for k, f in KEYS.items()}, index=oth.index)
print(f"computed keys for {len(oth):,} records in {time.time()-t0:.0f}s")

def explain_blocking(s1_id, show=6):
    rec = tr.loc[[s1_id]]
    truth = set(pairs.loc[pairs.s1 == s1_id, "m"])
    print(f"S1: {rec.business_name.iloc[0]!r} | {rec.business_address.iloc[0]!r}   true matches: {len(truth)}\n")
    union = set()
    for k, f in KEYS.items():
        key = f(rec).iloc[0]
        if key.endswith("|"):
            print(f"[{k}] key={key!r} -> empty key, block skipped"); continue
        block = oth_keys.index[oth_keys[k].values == key]
        hit = truth & set(block); union |= set(block)
        print(f"[{k}] key={key!r}\n    block size={len(block):,}   true matches caught={len(hit)}/{len(truth)}")
        sel = list(hit) + [x for x in block[:show + len(hit)] if x not in hit][:max(0, show - len(hit))]
        display(oth.loc[sel, ["business_name", "business_address"]].assign(is_match=lambda d: d.index.isin(truth)))
    print(f"UNION: {len(union):,} candidates, recall {len(truth & union)}/{len(truth)}")
    missed = truth - union
    if missed:
        print("missed by every key:"); display(tr.loc[list(missed), ["business_name", "business_address", "name_core", "addr_norm"]])

explain_blocking("S1-706225498")
""")

md(r"""
Read the output above: each key catches a *different* subset of the true matches, and some blocks contain
look-alikes that aren't matches (e.g. same street number, different business). That's fine — blocking only
has to *keep* true matches; the matcher later rejects the look-alikes.

Now a case where name blocking is dangerous — a **very common name**. The "exact name" block for a
"Meridian" business contains hundreds of different Meridians across the country:
""")

code(r"""
mer = s1[(s1.name_core == "meridian") & s1.index.isin(pairs.s1)].index[0]
explain_blocking(mer, show=8)
""")

md(r"""
### Recall vs. cost across the validation fold
Numbers from `reports/eda_stats.json` (computed by `python -m ber.analysis` on all 220k validation S1 entities;
blocks with >2000 records were skipped to keep it tractable).
""")

code(r"""
st = json.load(open(ROOT / "reports/eda_stats.json"))
b = pd.DataFrame(st["blocking_key_recall_val"]).T[["pair_recall_pct", "cands_per_s1"]].astype(float)
display(b)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.scatter(b.cands_per_s1, b.pair_recall_pct, s=60, color=[GRAY]*4 + [BLUE], zorder=3, edgecolor="#fcfcfb", linewidth=2)
for k, r in b.iterrows():
    ax.annotate(k, (r.cands_per_s1, r.pair_recall_pct), xytext=(6, -3), textcoords="offset points", fontsize=8.5)
ax.axhline(98, color=AQUA, lw=1.2, ls="--"); ax.text(5, 98.5, "target ≥98% recall", color="#52514e", fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("candidates per S1 (log)"); ax.set_ylabel("pair recall %")
ax.set_title("Simple exact keys: recall ceiling ~91% at 536 candidates/S1"); ax.set_ylim(35, 102); plt.show()
""")

md(r"""
**Why exact keys fall short here:** a single typo (`Cabrera 5ecure` fixed, but `Venmfes` isn't), a swapped
word, an invented brand name, or an empty address changes the key → the true match lands in a different block.
Meanwhile common names/streets make huge blocks. We need **fuzzy** blocking.

## 8. A better blocker: TF-IDF nearest neighbours (demo on one state)

Instead of "same key or nothing", represent each record as a vector of **character 3-grams** weighted by
TF-IDF (rare n-grams like `bre`,`cab` count more than `inc`, `ser`), and retrieve the **top-k most similar**
records by cosine similarity. A typo only changes a few n-grams, so similarity stays high.

We do this separately for the **name** and the **address**, and also a **combined** score (average of the two).
To keep it fast in a notebook we restrict to one US state (records whose state is missing are therefore
excluded here — a real pipeline also searches those).
""")

code(r"""
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

STATE, COUNTRY = "ky", "US"           # try "oh", "me", or an Indian code like "ka" with COUNTRY="India"
val_ids = set(split.source1_entity_id[split.fold == "val"])
Q = s1[(s1.country == COUNTRY) & (s1.addr_state == STATE) & s1.index.isin(val_ids)]
D = oth[(oth.country == COUNTRY) & (oth.addr_state == STATE)]
truth_pairs = pairs[pairs.s1.isin(Q.index)]
print(f"queries (val S1): {len(Q):,}   searchable S2/S3: {len(D):,}   true pairs: {len(truth_pairs):,}")
print(f"true pairs whose record is outside this state block (state missing/different): "
      f"{(~truth_pairs.m.isin(D.index)).mean():.1%}")

def tfidf(field):
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True, dtype=np.float32)
    v.fit(pd.concat([Q[field], D[field]]))
    return v.transform(Q[field]), v.transform(D[field])

t0 = time.time()
qn, dn = tfidf("name_norm"); qa, da = tfidf("addr_norm")
print(f"vectorized in {time.time()-t0:.1f}s; name vocab {qn.shape[1]:,}, addr vocab {qa.shape[1]:,}")
""")

code(r"""
KMAX = 50
def topk(sim_fn, k=KMAX, chunk=1000):
    out = np.empty((len(Q), k), dtype=np.int64)
    for i in range(0, len(Q), chunk):
        S = sim_fn(slice(i, i + chunk))
        S = S.toarray() if sparse.issparse(S) else S
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        order = np.take_along_axis(S, idx, 1).argsort(axis=1)[:, ::-1]
        out[i:i + chunk] = np.take_along_axis(idx, order, 1)
    return out

t0 = time.time()
nn = {
    "name only":      topk(lambda s: qn[s] @ dn.T),
    "address only":   topk(lambda s: qa[s] @ da.T),
    "name + address": topk(lambda s: (qn[s] @ dn.T + qa[s] @ da.T) * 0.5),
}
print(f"retrieval done in {time.time()-t0:.1f}s")

qpos = {q: i for i, q in enumerate(Q.index)}
t_q = truth_pairs.s1.map(qpos).values; t_m = truth_pairs.m.values
D_ids = D.index.values
ks = [1, 2, 3, 5, 10, 20, 30, 50]
rec = {}
for name, I in nn.items():
    cand_ids = D_ids[I]                                   # (n_queries, KMAX)
    rank = np.full(len(t_q), 10**9)
    for j in range(KMAX):
        hit = (cand_ids[t_q, j] == t_m) & (rank == 10**9)
        rank[hit] = j + 1
    rec[name] = [100 * (rank <= k).mean() for k in ks]
rec = pd.DataFrame(rec, index=pd.Index(ks, name="k (candidates per S1)")).round(2)
display(rec)

fig, ax = plt.subplots(figsize=(6.5, 3.6))
for (name, col) in zip(rec.columns, [BLUE, ORANGE, AQUA]):
    ax.plot(rec.index, rec[name], marker="o", ms=5, lw=2, color=col, label=name)
    ax.annotate(name, (rec.index[-1], rec[name].iloc[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8.5, va="center")
ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
ax.set_xlabel("top-k candidates per S1 (log)"); ax.set_ylabel("pair recall % (within state)")
ax.set_title(f"TF-IDF kNN blocking — {COUNTRY}/{STATE.upper()}"); ax.set_xlim(0.8, 120); ax.legend(loc="lower right"); plt.show()
""")

md(r"""
Compare with the exact-key union (~91% recall at **536** candidates): fuzzy retrieval on name+address should
reach the high 90s with only **tens** of candidates. The final pipeline will combine several retrievers
(name kNN, address kNN, combined kNN, maybe a multilingual embedding model) **per source** (top-k from S2 and
top-k from S3 separately, since an entity has up to 5–6 copies in each) plus cheap exact keys, and take the union.

Let's inspect what the retriever returns for one query — the matcher's job is to pick the true ones out of this list:
""")

code(r"""
qi = 7                                        # change me
I = nn["name + address"]
q = Q.iloc[qi]; truth = set(pairs.loc[pairs.s1 == Q.index[qi], "m"])
print(f"QUERY: {q.business_name!r} | {q.business_address!r}   ({len(truth)} true matches)")
cand = D.iloc[I[qi, :15]][["src", "business_name", "business_address"]].copy()
cand["score"] = ((qn[qi] @ dn[I[qi, :15]].T + qa[qi] @ da[I[qi, :15]].T) * 0.5).toarray().ravel().round(3)
cand["TRUE MATCH"] = cand.index.isin(truth)
cand
""")

# =============================================================================
md(r"""
## 9. France — the unseen test country

No France labels exist. The pipeline must treat `country` as an open label. What changes:
different legal forms (SARL, SAS, EURL…), French street words (RUE/R, AV, BD), departments instead of
regions in S2/S3, and very generic vocabulary (club, école, amicale).
""")

code(r"""
fr = te[te.country == "France"]
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
fr.name_legal.replace("", "<none>").value_counts().head(10)[::-1].plot.barh(ax=axs[0], color=AQUA, width=.75)
axs[0].set_title("France: legal forms found"); axs[0].set_xlabel("records")
fr.name_core.str.split().explode().value_counts().head(15)[::-1].plot.barh(ax=axs[1], color=AQUA, width=.75)
axs[1].set_title("France: most common name tokens (low information!)"); axs[1].set_xlabel("records")
plt.tight_layout(); plt.show()
fr.sample(10, random_state=0)[["src", "business_name", "business_address", "name_norm", "addr_norm", "addr_state"]]
""")

code(r"""
# Profile comparison: are cleaned fields populated similarly in France vs. the training countries?
tp = pd.DataFrame(st["train_country_profile"]).assign(split="train")
sp_ = pd.DataFrame(st["test_country_profile"]).assign(split="test")
pd.concat([tp, sp_]).set_index(["split", "src", "country"]).drop(columns="n")
""")

md(r"""
## Takeaways

- **Clean first**: normalization lifts true-pair name similarity from ~85 to ~93 (India S2: 71 → 94).
- **Constraints to exploit**: same-country only; each S2/S3 record → at most one S1.
- **Name alone is not enough** (collisions) → address and context features decide.
- **Blocking**: exact keys cap recall at ~91%; fuzzy TF-IDF / embedding kNN, per source, unioned → target ≥98% at ≤50 candidates.
- **France**: rely on country-agnostic features; watch calibration.

Next notebook: candidate generation at full scale + pairwise matcher.
""")

# =============================================================================
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "amazon-er", "display_name": "Python (amazon-er)", "language": "python"}
nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s) for t, s in cells]
out = Path(__file__).parent / "01_eda_preprocessing_blocking.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
