# Business Entity Resolution — Data Cleaning & EDA Report

_Generated 2026-09-25. Stats: `reports/eda_stats.json` (from `python -m ber.analysis`). Examples: `reports/*.csv`._

## 1. Dataset at a glance

| File | Rows | Countries |
|---|---:|---|
| train_source1 | 2,206,821 | US 1.32M · India 0.88M |
| train_source2 | 5,034,616 | US 3.02M · India 2.02M |
| train_source3 | 5,285,603 | US 3.17M · India 2.12M |
| train_ground_truth | 2,206,821 | one row per S1 entity |
| test_source1 | 1,732,544 | India 0.81M · US 0.66M · **France 0.26M** |
| test_source2 | 4,887,273 | India 2.31M · US 1.87M · France 0.70M |
| test_source3 | 5,082,316 | India 2.41M · US 1.95M · France 0.73M |

Integrity checks all pass. Every file parses with no malformed rows (row count = line count). All `entity_id`s are unique, and every ground-truth ID exists in the source files.

## 2. Ground-truth structure (what the matcher must output)

| Finding | Value | Implication |
|---|---|---|
| Matches per S1 entity | mean 3.46, median 3, max 11 | A typical entity has several matches, so recall matters even under F0.5. |
| Singletons (no match) | **5.6%** (identical for US/India) | The "predict nothing" baseline scores only about 0.056. |
| S2 matches per S1 | 0–5 (mode 1) | Sources 2 and 3 both contain **duplicates of the same business**. |
| S3 matches per S1 | 0–6 (mode 1) | |
| S2/S3 records matched to >1 S1 | **0** | Each S2/S3 record belongs to at most one S1 entity. That is a hard **1-to-many assignment constraint** usable for post-processing. |
| Cross-country matches | **0** | Blocking can be done strictly within country. |
| S2 / S3 records with no S1 match | 26.6% / 25.4% | About a quarter of the pool is **distractors** (see §5). |
| Singleton rate vs S1 features | flat ~5.6% across name length, legal form, country | Singletons can't be predicted from S1 alone. They must be decided by the absence of a good candidate. |

## 3. Noise catalogue (measured rates)

Source 1 is **clean and canonical**: title case, Latin script only, no junk, no empty fields. **All noise is in Sources 2 and 3**, and the two differ in style:

| Pattern | S2 (US / IN) | S3 (US / IN) | Example |
|---|---|---|---|
| Empty or null address (`NULL`, `N/A`, `<NULL>`, `null`) | 3.7 / 2.9% | 3.5 / 3.1% | |
| Name ALL-CAPS | 21.7 / 14.9% | 3.2 / 2.5% | `CABRERA SECURE SCIENCES` |
| Address ALL-CAPS | **89.7** / 23.8% | 0 / 0% | `709 HACKBERRY ST, TILDEN, TX` |
| Injected accents | 6.7 / 4.3% | 6.8 / 5.4% | `Seneca Ínc`, `Prívate` |
| OCR digit-for-letter | 2.7 / 2.0% | 2.7 / 2.1% | `5ecure`, `C0mpany` |
| Junk prefix / brackets / pipe | ~2 / ~10% | ~2 / ~10% | `-- Holloway…`, `(Limited)`, `\| www.x.com` |
| Web domain as the name | 4.6 / 3.5% | 4.4 / 3.8% | `emerakeystonecom`, `Reditprivate.Com` |
| Alias markers (dba, a/k/a, formerly, doing business as) | 0% | **2.7 / 1.8%** | `Jaxaria Formerly Cabrera Secure Sciences` |
| Name in an Indic script | — / **23%** | — / 12% | `एसएस फूड प्राइवेट लिमिटेड` (= "SS Food Private Limited") |
| Address has an Indic component | — / 24% | — / 22% | `महाराष्ट्र`, `ಕರ್ನಾಟಕ` |
| State format | code (`TX`) | full name (`Texas`) US; code (`MH`) IN | S1 uses a code for the US and the full name for India |
| House-number mangling | leading zeros 5%, ranges `5001-5003`, suffix `4514D`, dropped digit `344`↔`1344` | | |
| "CDP" suffix | 1.2% | 1.2% | `CHICAGOCDP`, `CHILLICTHE CDP` |
| Component reordering | common | common | `IL, Chicago, 315 80th St` |
| Generic word swap/insertion | | | `Indchem Power` → `Indchem Center / Services`, `Rays Office` → `Rays Center` |

Indic scripts in Sources 2 and 3 (India): Devanagari 13%, plus Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Oriya and Gurmukhi at 0.2–2% each. These are **word-by-word phonetic transliterations of the English name**, not translations.

**Train and test have the same noise profile** for US/India (all rates within ±0.5 pp). The only distribution shift is **France**:
- Legal forms: SARL, SAS, EURL, SA, SASU, SCI, EI, SNC.
- French street abbreviations: R/RUE, AV, BD.
- S1 gives the region; S2/S3 give the department (Nord, Gironde, Loire-Atlantique) or nothing.
- Only ~17 cities.
- Very generic name vocabulary (club, école, amicale, comité, maison, centre).
- French stopwords (de, du, des, la).

## 4. Cleaning pipeline (`src/ber/normalize.py`, `preprocess.py`)

Each record gets these derived fields. Raw columns are kept.

- **`name_norm`**
  - Text cleanup: NFKC, strip accents, lowercase, null tokens to empty, `&`/`+` → `and`, strip punctuation and brackets, collapse whitespace.
  - OCR repair inside alphabetic tokens (`0→o 1→l 3→e 4→a 5→s …`).
  - Web domains extracted to `name_domain` (for example `ipower.com` → `ipower`).
  - Alias split: `X dba/a/k/a/formerly Y` → `Y` is the name, `X` goes to `name_alias`. The text after the marker is the one that matches S1.
  - Indic tokens are transliterated through a **dictionary learned from training pairs**, falling back to `unidecode`.
- **`name_core` / `name_legal`**: legal forms are canonicalized and split out (Inc, LLC, Ltd, Pvt, Corp, Co, LLP, SARL, SAS, EURL, …). Stopwords are dropped.
- **`addr_norm`**: components are split on commas. Null components are dropped and state components are pulled out into **`addr_state`**. This covers US and India codes and full names, old names (Orissa, Keralam), French regions and departments, and **native-script state names learned from pairs**. Then:
  - Street-type canonicalization (street/saint→st, road/marg→rd, avenue/av→ave, rue→r, near→nr, …).
  - City aliases (Bombay→Mumbai, Calcutta→Kolkata, Bengaluru→Bangalore, Gurugram→Gurgaon, …).
  - Leading zeros stripped, `cdp` removed, unit/door designators dropped.
- **`addr_house`, `addr_numbers`, `addr_postcode`**: numeric keys.

**Learned resources**, fit on the **train fold only** so validation stays honest (`data/processed/resources.json`):
- 16 native-script state names, each mapped to its state code at ≥80% purity (`தமிழ்நாடு`→TN, `ಕರ್ನಾಟಕ`→KA, …).
- A 428-entry transliteration dictionary: phonetic skeleton → English token (`prvt`→private, `lmtd`→limited, `mrktng`→marketing, `lnjstks`→logistics, …).

**Validation split:** 10% of S1 entities (220,140), seed 42 → `data/processed/split.parquet`. The S2/S3 candidate pool is shared.

### Effect of cleaning on true-match similarity (300k sampled positive pairs)

| Metric (token-set ratio, 0–100) | Raw mean / p10 | Cleaned mean / p10 |
|---|---|---|
| Name | 84.5 / 43.5 | **93.2 / 80.0** |
| Address | 85.1 / 70.0 | **92.9 / 83.3** |
| India S2 names only | 71.3 (mean) | **93.5** (transliteration was the big gain) |

Random same-country non-matches score about 33 on both fields, so they separate easily. **The hard negatives are elsewhere** (§5).

## 5. Where the difficulty is

1. **Name collisions are pervasive.**
   - **49%** of S1 entities share their exact core name with at least one other S1 entity (for example `meridian` ×559, `summit`, `cedar`, `family center`).
   - Blocking on exact core name has only **3.5% precision**.
   - So the address must carry the decision. A name-only matcher will create false merges, and F0.5 punishes those twice as hard as misses.
2. **Distractors look real.** 25% of *unmatched* S2/S3 records have a name that also appears in S1 (vs 62% of matched ones). They are plausible businesses that should match nothing. Almost none of them have an empty address (0.3% vs 4.4% for matched records).
3. **Positive-pair taxonomy**, using a similarity cutoff of 50:
   - **92.4%**: name and address both similar (easy).
   - **3.0%**: address only. The name is an invented brand (`Nylaorbiquo`, `Orbidelta`) or a squashed domain (`reditprivate.com`, `#kuenzikristina`), so the address is identical. Comparing the S1 name *with spaces removed* against the domain rescues many of these.
   - **4.6%**: name only. The address is empty (~4.4% of matched records) or heavily truncated.
   - **0.02%**: both weak. The legal-form or generic word changed and the address is empty.
4. **House-number agreement is only 72%** on true pairs. Numbers get mangled (`344` vs `1344`, `5001-5003`), so use fuzzy number features rather than a hard key.
5. **Generic tails** like Services, Center, Partners, Enterprises and Group get appended or swapped. These tokens need low weight (IDF), otherwise `Indchem Center` looks like a different business from `Indchem Power`.

## 6. Blocking baselines (validation fold, pair recall)

| Key (within country) | Pair recall | Candidates / S1 |
|---|---:|---:|
| exact `name_core` | 59.3% | 58 |
| first name token + state | 75.4% | 293 (6.5% of S1 in oversize blocks, which were skipped) |
| house number + first street token | 48.7% | 73 |
| **union of simple keys** | **91.2%** | 536 |

Simple exact keys hit a ceiling around 91% with too many candidates. The recommended next step is **TF-IDF char n-gram ANN on name and address separately, plus multilingual embeddings** (for example `intfloat/multilingual-e5-small` or `BAAI/bge-m3`, both MIT-licensed), with per-source top-k. The target is >98% recall at ≤50 candidates per S1.

## 7. Recommendations for modelling

- **Block strictly within `country`**, and treat country as an open label so France flows through the same path.
- **Pairwise classifier (LightGBM)** on name features (core TSR, Jaccard, char-n-gram cosine, legal-form agreement, IDF-weighted overlap, domain-vs-squashed-name), address features (TSR, numeric-token overlap, house-number fuzzy match, state and postcode agreement, empty flags), per-source indicators, and **context features**: the candidate's rank and score gap among all S1 candidates for that record. Context features are what defend against name collisions.
- **Enforce the constraint**: assign each S2/S3 record to at most its single best-scoring S1 entity.
- **Tune the threshold directly on validation macro-F0.5** (singletons included), per country if needed. A slightly high threshold is favoured.
- **France has no labels.** Rely on country-agnostic features and add French stopwords and legal forms (already partly covered). Before submitting, inspect France score distributions against US/India for calibration drift.

## 8. Reproduce

```bash
conda activate amazon-er
export PYTHONPATH=code/business_entity_resolution/src
python -m ber.convert                 # dataset/*.tsv -> data/raw/*.parquet        (~2 min)
python -m ber.preprocess --workers 32 # split + resources + normalized parquet      (~10 min)
python -m ber.analysis                # reports/eda_stats.json + example CSVs        (~15 min)
```
