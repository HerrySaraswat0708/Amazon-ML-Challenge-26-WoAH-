# Business Entity Resolution — Amazon ML Challenge

Match Source 2 / Source 3 business records to each Source 1 entity. Metric: macro F0.5 per Source 1 entity.

## Environment

```bash
conda create -n amazon-er python=3.11 -y
conda activate amazon-er
pip install -r requirements.txt
```

## Data

Place the challenge files at the repo root:

```
dataset/train/train_source{1,2,3}.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source{1,2,3}.tsv
```

## Run

```bash
export PYTHONPATH=code/business_entity_resolution/src
python -m ber.convert                  # dataset/*.tsv -> data/raw/*.parquet
python -m ber.preprocess --workers 32  # val split, learned resources, normalized records -> data/processed/
python -m ber.analysis                 # EDA stats -> reports/eda_stats.json
```

_Blocking → matching → `output/*.tsv`: TBD._

Validate before submitting:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
