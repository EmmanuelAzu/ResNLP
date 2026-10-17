"""Framing: does the embedding neighbourhood of a word differ between Isigidimi and a
date-matched Imvo subsample by more than a permutation null allows?

Design
  * Window: 1 Nov 1884 - 1888 (Imvo exists; Isigidimi has no 1886 issues).
  * D independent date-matched Imvo subsamples (same months, same token count as Isigidimi).
  * Real comparison, per draw d: fastText(Isigidimi) vs fastText(Imvo draw d).
  * Null, per draw d: pool Isigidimi issues and the Imvo draw's issues, reassign them at
    issue level to two groups of equal size within strata (Nov84-85 / 1887 / 1888), train on
    each group, measure the same shift. K permutations per draw (D*K in total).
  * Shift(w) = 1 - Jaccard of the top-10 neighbour sets (neighbours searched within the
    shared vocabulary V, so the two spaces need no alignment).
  * Per-word z against that word's own null; p from the pooled null z distribution; BH-FDR.
  * Frequency control: the permuted groups have equal word frequencies but the real ones do
    not (e.g. Imvo uses 'palamente' more), so each word is also ranked against 50 words
    matched on mean frequency and frequency ratio (excluding the theory/control words).
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import CONTROL, OUT, THEORY, count_tokens, draw_imvo_subsample, flatten, load_pages
from emb import DEFAULT_CFG, jaccard_shift, neighbours, train, variant_share, vectors

MIN_V, MIN_P, K_NB = 20, 25, 10


def concept(word):
    for group, table in (("theory", THEORY), ("control", CONTROL)):
        for name, stems in table.items():
            if any(s in word for s in stems):
                return group, name
    return None, None


def units_by_issue(df):
    return {k: [s for ss in g.sents for s in ss] for k, g in df.groupby(["paper", "date"])}


def stratum(date):
    return "84-85" if date < "1886" else date[:4]


def permute(units, rng):
    """Random issue-level split into two token-balanced groups within strata."""
    by = {}
    for key, ss in units.items():
        by.setdefault(stratum(key[1]), []).append((key, ss, sum(map(len, ss))))
    A, B = [], []
    for items in by.values():
        order = rng.permutation(len(items))
        ta = tb = 0
        for i in order:
            _, ss, n = items[i]
            if ta <= tb:
                A.extend(ss); ta += n
            else:
                B.extend(ss); tb += n
    return A, B


def run_job(args):
    sents, seed = args
    return vectors(train(sents, seed, CFG), V_WORDS)


def init_worker(cfg, v_words):
    global CFG, V_WORDS
    CFG, V_WORDS = cfg, v_words


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--perms-per-draw", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="2 draws x 2 permutations (smoke test)")
    args = ap.parse_args()
    D, K = (2, 2) if args.quick else (args.draws, args.perms_per_draw)

    cfg = {**DEFAULT_CFG}
    hp = OUT / "hparams.json"
    if hp.exists():
        cfg.update(json.loads(hp.read_text()))
    print("config:", cfg)

    pages = load_pages()
    isig_pages = pages[pages.paper == "ISIG"]
    isig_units = units_by_issue(isig_pages)
    isig_sents = flatten(isig_pages)
    c_isig = count_tokens(isig_sents)

    draws = [draw_imvo_subsample(pages, np.random.default_rng(1000 + d)) for d in range(D)]
    c_imvo = [count_tokens(flatten(s)) for s in draws]
    print("Isigidimi tokens", sum(c_isig.values()), "| Imvo draws", [int(s.n_tok.sum()) for s in draws])

    # shared vocabulary and probe words (defined from Isigidimi + the average Imvo draw)
    c_imvo_mean = {w: np.mean([c.get(w, 0) for c in c_imvo]) for c in c_imvo[:1] for w in c}
    words = [w for w in c_isig if c_isig[w] >= MIN_V and c_imvo_mean.get(w, 0) >= MIN_V]
    words.sort()
    probe = np.array([i for i, w in enumerate(words) if min(c_isig[w], c_imvo_mean[w]) >= MIN_P])
    print(f"|V|={len(words)}  |P|={len(probe)}")

    rng = np.random.default_rng(7)
    jobs, meta = [], []
    for d in range(D):
        jobs.append((isig_sents, 100 + d)); meta.append(("real_isig", d))
        jobs.append((flatten(draws[d]), 200 + d)); meta.append(("real_imvo", d))
        units = {**isig_units, **units_by_issue(draws[d])}
        for k in range(K):
            A, B = permute(units, rng)
            jobs.append((A, 3000 + 10 * (d * K + k))); meta.append(("perm_a", d, k, sum(map(len, A))))
            jobs.append((B, 3001 + 10 * (d * K + k))); meta.append(("perm_b", d, k, sum(map(len, B))))
    print(len(jobs), "models to train")
    with ProcessPoolExecutor(4, initializer=init_worker, initargs=(cfg, words)) as ex:
        mats = list(ex.map(run_job, jobs, chunksize=1))
    nb = [neighbours(np.nan_to_num(m), K_NB) for m in mats]

    mat = {m: i for i, m in enumerate(meta)}
    real = np.array([jaccard_shift(nb[mat[("real_isig", d)]], nb[mat[("real_imvo", d)]]) for d in range(D)])
    null, bal = [], []
    for d in range(D):
        for k in range(K):
            ia = [i for i, m in enumerate(meta) if m[:3] == ("perm_a", d, k)][0]
            null.append(jaccard_shift(nb[ia], nb[ia + 1]))
            bal.append(meta[ia][3] / max(meta[ia + 1][3], 1))
    null = np.array(null)                            # (D*K, |V|)
    R, N = real[:, probe], null[:, probe]            # probe words only

    # ---- global test: mean shift over probe words vs permutation distribution
    null_mean = N.mean(1)
    glob = [{"draw": d, "obs_mean_shift": float(R[d].mean()),
             "p": float((1 + (null_mean >= R[d].mean()).sum()) / (len(null_mean) + 1))} for d in range(D)]
    # ---- per-word z against own null, p from pooled null z
    mu, sd = N.mean(0), np.maximum(N.std(0, ddof=1), 0.05)
    z_obs = ((R - mu) / sd)
    z_null = ((N - mu) / sd).ravel()
    pooled = np.sort(z_null)
    p_w = np.array([[(1 + (len(pooled) - np.searchsorted(pooled, z, side="left"))) / (len(pooled) + 1)
                     for z in zrow] for zrow in z_obs])
    q_w = np.array([bh(p) for p in p_w])

    # ---- frequency-matched control rank
    pw = [words[i] for i in probe]
    fi = np.array([c_isig[w] for w in pw], float)
    fm = np.array([c_imvo_mean[w] for w in pw], float)
    feat = np.c_[np.log10((fi + fm) / 2), np.abs(np.log2(fm / fi))]
    feat = (feat - feat.mean(0)) / feat.std(0)
    kind = np.array([concept(w)[0] is not None for w in pw])
    T = R.mean(0)
    matched = np.full(len(pw), np.nan)
    for i in range(len(pw)):
        cand = np.where(~kind & (np.arange(len(pw)) != i))[0]
        near = cand[np.argsort(((feat[cand] - feat[i]) ** 2).sum(1))[:50]]
        matched[i] = (T[near] < T[i]).mean()

    rows = pd.DataFrame({
        "word": pw, "group": [concept(w)[0] for w in pw], "concept": [concept(w)[1] for w in pw],
        "n_isig": fi, "n_imvo_draw": fm, "log2_ratio": np.log2(fm / fi),
        "shift_obs": T, "shift_null_mean": mu, "shift_null_sd": sd,
        "z_mean": z_obs.mean(0), "q_median_draw": np.median(q_w, 0),
        "share_draws_q<0.05": (q_w < 0.05).mean(0), "matched_percentile": matched})
    rows.to_csv(OUT / "framing_words.csv", index=False)

    summary = {"config": cfg, "D": D, "K": K, "n_V": len(words), "n_P": len(probe),
               "global_tests": glob, "null_mean_shift": [float(null_mean.mean()), float(null_mean.std())],
               "real_mean_shift_by_draw": [float(r.mean()) for r in R],
               "perm_group_size_ratio_range": [float(min(bal)), float(max(bal))],
               "variant_neighbour_share_isig": variant_share(words, nb[mat[("real_isig", 0)]], probe),
               "variant_neighbour_share_imvo": variant_share(words, nb[mat[("real_imvo", 0)]], probe),
               "spearman_shift_vs_abs_logratio": float(pd.Series(T).corr(pd.Series(np.abs(rows.log2_ratio)), method="spearman")),
               "n_words_sig_all_draws": int((q_w < 0.05).all(0).sum())}
    (OUT / "framing_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(OUT / "framing_raw.npz", real=real, null=null, words=np.array(words), probe=probe)
    print(json.dumps(summary, indent=2))
    print(rows[rows.group.notna()].sort_values(["group", "concept"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
