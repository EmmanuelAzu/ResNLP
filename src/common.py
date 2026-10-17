"""Shared helpers: cleaning, loading, the strict overlap window and Imvo subsampling."""
import gzip
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
PAGES_CSV = DATA / "pages.csv.gz"

# Imvo's first issue is 1884-11-03, so the shared period starts on 1 Nov 1884.
# Isigidimi has no 1886 issues in the archive (the masthead numbering shows it
# was published), so the matched months below are taken from Isigidimi itself.
WINDOW = ("1884-11-01", "1888-12-31")

TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n")


def clean_to_sentences(text: str):
    """Normalise a page and return a list of token lists (one per sentence)."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[‘’ʼ`]", "'", text)
    text = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", text)      # line-end hyphens
    text = text.lower()
    sents = []
    for chunk in SENT_SPLIT_RE.split(text):
        chunk = chunk.replace("\n", " ")
        toks = TOKEN_RE.findall(chunk)
        if toks:
            sents.append(toks)
    return sents


def load_pages(window=WINDOW) -> pd.DataFrame:
    """Pages as a DataFrame: paper, date, page, sents (list of token lists), n_tok."""
    df = pd.read_csv(PAGES_CSV)
    df["sents"] = df["text"].fillna("").map(lambda t: [s.split() for s in t.split("\n") if s])
    df["n_tok"] = df["sents"].map(lambda ss: sum(len(s) for s in ss))
    df["ym"] = df["date"].str[:7]
    if window:
        df = df[(df["date"] >= window[0]) & (df["date"] <= window[1])]
    return df.reset_index(drop=True)


def count_tokens(sents) -> Counter:
    c = Counter()
    for s in sents:
        c.update(s)
    return c


def flatten(pages: pd.DataFrame):
    return [s for ss in pages["sents"] for s in ss]


def draw_imvo_subsample(pages: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Date-matched Imvo subsample.

    For every year-month in which Isigidimi has text, draw random Imvo pages from the
    same month until the Imvo token count equals Isigidimi's for that month; the last
    page is truncated at a sentence boundary so the match is exact to one sentence.
    Returns a DataFrame of units (paper, date, page, sents, n_tok).
    """
    isig = pages[pages.paper == "ISIG"]
    imvo = pages[pages.paper == "IMVO"]
    target = isig.groupby("ym")["n_tok"].sum()
    out = []
    for ym, tgt in target.items():
        pool = imvo[imvo.ym == ym]
        if pool.empty:
            continue
        got = 0
        for i in rng.permutation(len(pool)):
            row = pool.iloc[i]
            sents = []
            for s in row.sents:
                if got >= tgt:
                    break
                sents.append(s)
                got += len(s)
            if sents:
                out.append({"paper": "IMVO", "date": row.date, "page": row.page, "ym": ym,
                            "sents": sents, "n_tok": sum(len(s) for s in sents)})
            if got >= tgt:
                break
    return pd.DataFrame(out)


def normalise_rows(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)


# Concept stems (matched as substrings of surface forms; old orthography has no aspiration
# marks, e.g. umteto not umthetho, rulumente not rhulumente).
THEORY = {"parliament": ["palamente"], "vote": ["voti"], "government": ["rulumente"],
          "rule": ["mbuso"], "law": ["mteto", "nteto"], "rights": ["lungelo"],
          "country": ["lizwe"], "land": ["mhlaba"], "freedom": ["kululek"], "humanity": ["buntu"]}
CONTROL = {"time": ["xesha"], "house": ["ndlu"], "rain": ["mvula"], "cattle": ["nkomo"]}
