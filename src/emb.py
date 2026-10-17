"""Embedding helpers: training, neighbour retrieval, shift measures, variant detection."""
import difflib

import numpy as np
from gensim.models import FastText, Word2Vec

from common import normalise_rows

DEFAULT_CFG = dict(kind="ft", vector_size=100, window=5, epochs=10, min_count=5,
                   min_n=3, max_n=6, bucket=200000, negative=5, sg=1)


def train(sents, seed, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    kind = cfg["kind"]
    common = dict(vector_size=cfg["vector_size"], window=cfg["window"], epochs=cfg["epochs"],
                  min_count=cfg["min_count"], negative=cfg["negative"], sg=cfg["sg"],
                  seed=seed, workers=1)
    if kind == "ft":
        return FastText(sents, min_n=cfg["min_n"], max_n=cfg["max_n"], bucket=cfg["bucket"], **common)
    return Word2Vec(sents, **common)


def vectors(model, words):
    """Unit vectors for `words`. fastText builds OOV words from subwords; word2vec
    words missing from the vocabulary come back as NaN rows."""
    rows = []
    for w in words:
        try:
            rows.append(model.wv[w])
        except KeyError:
            rows.append(np.full(model.vector_size, np.nan, dtype=np.float32))
    return normalise_rows(np.asarray(rows, dtype=np.float32))


def neighbours(mat, k=10):
    """Top-k cosine neighbours (indices) of every row among all rows, excluding itself."""
    sims = mat @ mat.T
    np.fill_diagonal(sims, -np.inf)
    sims = np.nan_to_num(sims, nan=-np.inf)
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    return idx


def jaccard_shift(nb_a, nb_b):
    """1 - Jaccard overlap of the neighbour sets of each word, between two models."""
    out = np.empty(len(nb_a))
    for i, (a, b) in enumerate(zip(nb_a, nb_b)):
        a, b = set(a), set(b)
        out[i] = 1.0 - len(a & b) / len(a | b)
    return out


def is_variant(w1, w2, min_len=5, frac=0.6):
    """Crude spelling/morphological-variant test: a long shared substring."""
    m = difflib.SequenceMatcher(None, w1, w2, autojunk=False).find_longest_match(0, len(w1), 0, len(w2))
    return m.size >= min_len and m.size >= frac * min(len(w1), len(w2))


def variant_share(words, nb, probe_idx):
    tot = hit = 0
    for i in probe_idx:
        for j in nb[i]:
            tot += 1
            hit += is_variant(words[i], words[j])
    return hit / max(tot, 1)
