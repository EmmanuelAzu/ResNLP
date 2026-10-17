"""Tune embedding hyper-parameters on text OUTSIDE the test window.

There is no intrinsic benchmark for historical isiXhosa, so the validation objective is
seed stability: mean top-10 neighbour Jaccard between two runs that differ only in seed,
on random probe words (freq >= 50). Tuning corpora (287k tokens each, whole issues):
  - Isigidimi 1876-1883   - Imvo 1889-1894
The test window (Nov 1884-1888) is never used here.
Also runs a word2vec ablation at the chosen setting and reports how much of the
neighbourhood is just spelling/morphological variants.
"""
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import OUT, count_tokens, flatten, load_pages
from emb import DEFAULT_CFG, jaccard_shift, neighbours, train, variant_share, vectors

TARGET = 287_000
GRID = dict(vector_size=[50, 100, 200], window=[3, 5, 8], epochs=[10, 20])
# Second round (--extend): the first round's best setting (dim 200, 20 epochs) was at the edge of
# the grid in two dimensions, so the grid is extended past it and the rows are merged.
GRID_EXT = dict(vector_size=[300, 400], window=[1], epochs=[30])   # round 5: epochs 30 at the final window (20 was best at windows 3 and 5)
SEEDS = (1, 2)
N_PROBE = 300


def make_corpus(pages, paper, lo, hi, rng):
    sub = pages[(pages.paper == paper) & pages.date.str[:4].astype(int).between(lo, hi)]
    issues = sub.groupby("date")
    order = rng.permutation(sub.date.unique())
    sents, n = [], 0
    for d in order:
        for ss in issues.get_group(d).sents:
            sents.extend(ss)
        n = sum(len(s) for s in sents)
        if n >= TARGET:
            break
    return sents


def run_one(args):
    corpus_id, cfg, seed = args
    sents = CORPORA[corpus_id]
    model = train(sents, seed, cfg)
    return vectors(model, WORDS[corpus_id])


def stability(mats):
    nbs = [neighbours(m) for m in mats]
    return 1.0 - jaccard_shift(nbs[0], nbs[1]).mean(), nbs


if __name__ == "__main__":
    pages = load_pages(window=None)
    rng = np.random.default_rng(0)
    CORPORA = {"isig_1876_83": make_corpus(pages, "ISIG", 1876, 1883, rng),
               "imvo_1889_94": make_corpus(pages, "IMVO", 1889, 1894, rng)}
    WORDS, PROBE_IDX = {}, {}
    for cid, sents in CORPORA.items():
        c = count_tokens(sents)
        WORDS[cid] = [w for w, n in c.items() if n >= 30]
        probe = [i for i, w in enumerate(WORDS[cid]) if c[w] >= 50]
        PROBE_IDX[cid] = rng.choice(probe, size=min(N_PROBE, len(probe)), replace=False)
        print(cid, sum(map(len, sents)), "tokens, vocab(>=30)", len(WORDS[cid]), "probe", len(PROBE_IDX[cid]))

    extend = "--extend" in sys.argv
    grid = GRID_EXT if extend else GRID
    configs = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    prev = pd.read_csv(OUT / "tuning_grid.csv") if extend else None
    if extend:        # skip settings already evaluated in round 1
        seen = {(r.vector_size, r.window, r.epochs) for r in prev.itertuples()}
        configs = [c for c in configs if (c["vector_size"], c["window"], c["epochs"]) not in seen]
        print("new settings:", len(configs))
    jobs_args = [(cid, cfg, s) for cfg in configs for cid in CORPORA for s in SEEDS]
    with ProcessPoolExecutor(4) as ex:
        mats = list(ex.map(run_one, jobs_args, chunksize=1))
    res, k = [], 0
    for cfg in configs:
        row = dict(cfg)
        scores = []
        for cid in CORPORA:
            pair = [mats[k], mats[k + 1]]
            k += 2
            idx = PROBE_IDX[cid]
            nbs = [neighbours(m) for m in pair]
            jac = 1.0 - jaccard_shift(nbs[0][idx], nbs[1][idx])
            row[f"stab_{cid}"] = jac.mean()
            scores.append(jac.mean())
        row["stability"] = float(np.mean(scores))
        res.append(row)
    df = pd.DataFrame(res)
    if extend:
        df = pd.concat([prev, df], ignore_index=True)
    df = df.sort_values("stability", ascending=False)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "tuning_grid.csv", index=False)
    print(df.round(3).to_string(index=False))
    best = {k_: int(v) for k_, v in df.iloc[0][["vector_size", "window", "epochs"]].items()}
    cfg = {**DEFAULT_CFG, **best}

    # word2vec ablation at the chosen setting (no subword information)
    ab = []
    for kind in ("ft", "w2v"):
        c2 = {**cfg, "kind": kind}
        args = [(cid, c2, s) for cid in CORPORA for s in SEEDS]
        with ProcessPoolExecutor(4) as ex:
            m = list(ex.map(run_one, args, chunksize=1))
        for i, cid in enumerate(CORPORA):
            pair = m[2 * i:2 * i + 2]
            nbs = [neighbours(np.nan_to_num(x)) for x in pair]
            idx = PROBE_IDX[cid]
            ab.append(dict(model=kind, corpus=cid,
                           stability=float(1 - jaccard_shift(nbs[0][idx], nbs[1][idx]).mean()),
                           variant_share=variant_share(WORDS[cid], nbs[0], idx)))
    ab = pd.DataFrame(ab)
    ab.to_csv(OUT / "ablation_ft_vs_w2v.csv", index=False)
    print(ab.round(3).to_string(index=False))
    cfg["kind"] = "ft"
    (OUT / "hparams.json").write_text(json.dumps(cfg, indent=2))
    print("chosen:", cfg)
