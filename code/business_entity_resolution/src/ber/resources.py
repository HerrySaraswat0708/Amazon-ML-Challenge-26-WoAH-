"""Fit data-driven normalization resources from labelled training pairs.

- native_state: native-script address component (e.g. 'महाराष्ट्र') -> state code,
  learned by co-occurrence with the Source 1 record's (Latin) state.
- native_token: phonetic skeleton of a transliterated Indic token -> English token,
  learned by positional alignment of equal-length name pairs.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from unidecode import unidecode

from .normalize import INDIC_RE, STATE_LOOKUP, Resources, clean_text, fold, squash_phonetic


def _majority(counts: dict[str, Counter], min_count: int, min_purity: float) -> dict[str, str]:
    out = {}
    for k, c in counts.items():
        tot = sum(c.values())
        v, n = c.most_common(1)[0]
        if n >= min_count and n / tot >= min_purity:
            out[k] = v
    return out


def s1_state(addr: str) -> str | None:
    for c in addr.split(","):
        code = STATE_LOOKUP.get(unidecode(c).lower().strip())
        if code:
            return code
    return None


def fit_resources(pairs: pd.DataFrame, min_count: int = 3) -> Resources:
    """pairs: columns s1_name, s1_addr, m_name, m_addr (matched S1 / S2|S3 records)."""
    st_counts: dict[str, Counter] = defaultdict(Counter)
    tok_counts: dict[str, Counter] = defaultdict(Counter)

    addr = pairs[pairs.m_addr.str.contains(INDIC_RE, regex=True)]
    for s1a, ma in zip(addr.s1_addr, addr.m_addr):
        code = s1_state(s1a)
        if not code:
            continue
        for c in ma.split(","):
            c = c.strip()
            if INDIC_RE.search(c):
                st_counts[c][code] += 1

    nm = pairs[pairs.m_name.str.contains(INDIC_RE, regex=True)]
    for s1n, mn in zip(nm.s1_name, nm.m_name):
        a = clean_text(fold(s1n)).split()
        b = [squash_phonetic(unidecode(t).lower()) for t in mn.split()]
        if len(a) != len(b):
            continue
        for x, y in zip(a, b):
            if y:
                tok_counts[y][x] += 1

    return Resources(
        native_state=_majority(st_counts, min_count=5, min_purity=0.8),
        native_token=_majority(tok_counts, min_count=min_count, min_purity=0.6),
    )


def save_resources(res: Resources, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"native_state": res.native_state, "native_token": res.native_token}, f,
                  ensure_ascii=False, indent=0)


def load_resources(path: str | Path) -> Resources:
    with open(path) as f:
        d = json.load(f)
    return Resources(**d)
