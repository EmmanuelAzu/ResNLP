"""Framing: do words keep the same company in Isigidimi and in a date-matched Imvo subsample?

Design
  * Window 1 Nov 1884 - 1888; D Imvo draws, each matched month-by-month to Isigidimi's size.
  * Real comparison per draw d: fastText(Isigidimi, seed 100+d) vs fastText(Imvo draw d).
  * Null per draw: Isigidimi + Imvo-draw issues pooled, re-assigned at issue level to two
    token-balanced groups (strata Nov84-85 / 1887 / 1888), one model per group; K per draw.
  * The same statistics are computed for every real pair and every null pair.

Measures (all alignment-free; the two spaces are trained independently)
  PRIMARY   profile shift  = 1 - Pearson r between a word's cosine profile over G (the 500 most
            frequent words shared by both papers, length > 2, mastheads removed) in the two spaces
            (the local-neighbourhood measure of Hamilton et al. 2016).  Continuous.
  ROBUSTNESS Jaccard shift = 1 - Jaccard of the top-10 neighbour sets.
  DIRECTIONAL (exploratory) D(w) = z(mean cos to religious anchors) - z(mean cos to political anchors),
            z-scored over the vocabulary within each space; Delta(w) = D_Isigidimi - D_Imvo.
            Anchors were fixed beforehand in outputs/anchors.json (08_anchors.py).

Inference: statistics are averaged over the D real draws; the matching null is the average of D
values resampled from the D*K null pairs (2000 resamples), so real and null have the same
averaging.  Per-word z and BH-FDR as before, plus a frequency-matched rank for each word.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import CONTROL, OUT, THEORY, apply_merges, count_tokens, draw_imvo_subsample, flatten, learn_merges, load_pages
from emb import DEFAULT_CFG, jaccard_shift, neighbours, train, variant_share, vectors

MIN_V, MIN_P, K_NB, N_ANCHOR = 20, 25, 10, 500
MASTHEAD = {"isigidimi", "sigidimi", "samaxosa", "xosa", "xhosa", "imvo", "mvo", "zabantsundu", "ndu",
            "ontsundu", "lovedale", "ngolwesi", "qonce", "komani", "bhayi", "town", "sir", "mr"}
SEED_STEMS = ("tixo", "kristu", "palamente", "voti")      # excluded as directional targets
NRESAMPLE = 2000


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
    by = {}
    for key, ss in units.items():
        by.setdefault(stratum(key[1]), []).append((ss, sum(map(len, ss))))
    A, B = [], []
    for items in by.values():
        ta = tb = 0
        for i in rng.permutation(len(items)):
            ss, n = items[i]
            if ta <= tb:
                A.extend(ss); ta += n
            else:
                B.extend(ss); tb += n
    return A, B


# ---- worker -------------------------------------------------------------------------------
def init_worker(cfg, words, g_idx, rel_idx, pol_idx):
    global CFG, WORDS, G_IDX, REL, POL
    CFG, WORDS, G_IDX, REL, POL = cfg, words, g_idx, rel_idx, pol_idx


def mean_cos_excluding_self(M, anchor_idx, n_rows):
    sims = M[:n_rows] @ M[anchor_idx].T                      # (n_rows, |anchors|)
    for j, a in enumerate(anchor_idx):
        if a < n_rows:
            sims[a, j] = np.nan
    return np.nanmean(sims, axis=1)


def run_job(args):
    sents, seed, n_v = args
    M = vectors(train(sents, seed, CFG), WORDS)
    M = np.nan_to_num(M)
    prof = M[:n_v] @ M[G_IDX].T                              # cosine profile over general anchors
    for j, a in enumerate(G_IDX):
        prof[a, j] = np.nan                                  # a word is not its own anchor
    z = lambda x: (x - x.mean()) / x.std()
    d_dir = z(mean_cos_excluding_self(M, REL, n_v)) - z(mean_cos_excluding_self(M, POL, n_v))
    nb = neighbours(M[:n_v], K_NB).astype(np.int16)
    return {"prof": prof.astype(np.float32), "dir": d_dir.astype(np.float32), "nb": nb}


# ---- statistics ---------------------------------------------------------------------------
def profile_shift(pa, pb):
    """1 - Pearson r over the anchor columns, ignoring each word's own column (NaN)."""
    m = ~(np.isnan(pa) | np.isnan(pb))
    a, b = np.where(m, pa, 0.0), np.where(m, pb, 0.0)
    n = m.sum(1)
    ma, mb = a.sum(1) / n, b.sum(1) / n
    da, db = (a - ma[:, None]) * m, (b - mb[:, None]) * m
    return 1.0 - (da * db).sum(1) / np.sqrt((da ** 2).sum(1) * (db ** 2).sum(1))


def bh(p):
    p = np.asarray(p); order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    out = np.empty_like(q); out[order] = np.minimum(q, 1)
    return out


def averaged_null(null, D, rng, n=NRESAMPLE):
    """Distribution of the mean of D null values (resampled from the pooled null pairs)."""
    idx = rng.integers(0, len(null), (n, D))
    return null[idx].mean(1)


def test_measure(real, null, rng):
    """real: (D, n) per-draw statistics, null: (J, n). Returns per-word table and global test."""
    D = real.shape[0]
    T = real.mean(0)
    avg = averaged_null(null, D, rng)
    mu, sd = avg.mean(0), np.maximum(avg.std(0, ddof=1), 1e-9)
    z = (T - mu) / sd
    zn = ((avg - mu) / sd).ravel()
    zs = np.sort(zn)
    p = (1 + (len(zs) - np.searchsorted(zs, z, side="left"))) / (len(zs) + 1)
    glob_obs, glob_null = T.mean(), avg.mean(1)
    g = {"obs": float(glob_obs), "null_mean": float(glob_null.mean()), "null_sd": float(glob_null.std(ddof=1)),
         "z": float((glob_obs - glob_null.mean()) / glob_null.std(ddof=1)),
         "p": float((1 + (glob_null >= glob_obs).sum()) / (NRESAMPLE + 1))}
    return T, mu, sd, z, p, bh(p), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--perms-per-draw", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--window", type=int, default=None, help="override the tuned window (sensitivity run)")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _w5")
    ap.add_argument("--segnorm", action="store_true", help="join disjunctively written forms (uku ba -> ukuba) before anything else")
    ap.add_argument("--drop-anchor", action="append", default=[], help="remove this word from the directional anchors (robustness check)")
    args = ap.parse_args()
    D, K = (2, 2) if args.quick else (args.draws, args.perms_per_draw)

    cfg = {**DEFAULT_CFG}
    if (OUT / "hparams.json").exists():
        cfg.update(json.loads((OUT / "hparams.json").read_text()))
    if args.window:
        cfg["window"] = args.window
    print("config:", cfg)
    anchors = json.loads((OUT / "anchors.json").read_text())
    for w in args.drop_anchor:
        for side in ("religious", "political"):
            anchors[side] = [x for x in anchors[side] if x != w]
    if args.drop_anchor:
        print("dropped anchors:", args.drop_anchor)

    pages = load_pages()
    tag = args.tag + ("_segnorm" if args.segnorm else "")
    if args.segnorm:
        merges, _, _ = learn_merges(load_pages(window=None))
        pages["sents"] = pages.sents.map(lambda ss: apply_merges(ss, merges))
        pages["n_tok"] = pages.sents.map(lambda ss: sum(len(x) for x in ss))
        print(f"segmentation normalisation: {len(merges)} merge rules")
    isig_pages = pages[pages.paper == "ISIG"]
    isig_units, isig_sents = units_by_issue(isig_pages), flatten(isig_pages)
    c_isig = count_tokens(isig_sents)
    draws = [draw_imvo_subsample(pages, np.random.default_rng(1000 + d)) for d in range(D)]
    c_imvo = [count_tokens(flatten(s)) for s in draws]
    c_mean = {w: np.mean([c.get(w, 0) for c in c_imvo]) for w in c_imvo[0]}

    words = sorted(w for w in c_isig if c_isig[w] >= MIN_V and c_mean.get(w, 0) >= MIN_V)
    n_v = len(words)
    probe = np.array([i for i, w in enumerate(words) if min(c_isig[w], c_mean[w]) >= MIN_P])
    freq = {w: min(c_isig[w], c_mean[w]) for w in words}
    g_words = [w for w in sorted(words, key=lambda w: -freq[w]) if len(w) > 2 and w not in MASTHEAD][:N_ANCHOR]
    extra = [w for w in anchors["religious"] + anchors["political"] if w not in set(words)]
    all_words = words + sorted(set(extra))
    ix = {w: i for i, w in enumerate(all_words)}
    g_idx = np.array([ix[w] for w in g_words])
    rel_idx = np.array([ix[w] for w in anchors["religious"]])
    pol_idx = np.array([ix[w] for w in anchors["political"]])
    print(f"|V|={n_v} |P|={len(probe)} |G|={len(g_words)} rel={len(rel_idx)} pol={len(pol_idx)}")

    rng = np.random.default_rng(7)
    jobs, meta = [], []
    for d in range(D):
        jobs.append((isig_sents, 100 + d, n_v)); meta.append(("ri", d))
        jobs.append((flatten(draws[d]), 200 + d, n_v)); meta.append(("rm", d))
        units = {**isig_units, **units_by_issue(draws[d])}
        for k in range(K):
            A, B = permute(units, rng)
            jobs.append((A, 3000 + 10 * (d * K + k), n_v)); meta.append(("pa", d, k))
            jobs.append((B, 3001 + 10 * (d * K + k), n_v)); meta.append(("pb", d, k))
    print(len(jobs), "models")
    with ProcessPoolExecutor(4, initializer=init_worker, initargs=(cfg, all_words, g_idx, rel_idx, pol_idx)) as ex:
        res = list(ex.map(run_job, jobs, chunksize=1))
    at = {m: r for m, r in zip(meta, res)}

    # per-pair statistics: profile shift, Jaccard shift, directional delta
    def pair(a, b):
        return (profile_shift(a["prof"], b["prof"]),
                jaccard_shift(a["nb"], b["nb"]),
                a["dir"] - b["dir"])
    real = [pair(at[("ri", d)], at[("rm", d)]) for d in range(D)]
    null = [pair(at[("pa", d, k)], at[("pb", d, k)]) for d in range(D) for k in range(K)]
    stat = {name: (np.array([r[i] for r in real]), np.array([r[i] for r in null]))
            for i, name in enumerate(["profile", "jaccard", "delta"])}

    rs = np.random.default_rng(11)
    pw = [words[i] for i in probe]
    fi = np.array([c_isig[w] for w in pw], float); fm = np.array([c_mean[w] for w in pw], float)
    group = [concept(w)[0] for w in pw]
    table = pd.DataFrame({"word": pw, "group": group, "concept": [concept(w)[1] for w in pw],
                          "n_isig": fi, "n_imvo_draw": fm, "log2_ratio": np.log2(fm / fi)})
    summary = {"config": cfg, "D": D, "K": K, "n_V": n_v, "n_P": len(probe), "n_anchors": len(g_words),
               "anchors_religious": anchors["religious"], "anchors_political": anchors["political"]}
    feat = np.c_[np.log10((fi + fm) / 2), np.abs(np.log2(fm / fi))]
    feat = (feat - feat.mean(0)) / feat.std(0)
    kind = np.array([g is not None for g in group])
    concept_rows = []
    for name in ("profile", "jaccard"):
        r_, n_ = stat[name]
        T, mu, sd, z, p, q, g = test_measure(r_[:, probe], n_[:, probe], rs)
        table[f"{name}_shift"], table[f"{name}_null_mean"], table[f"{name}_z"] = T, mu, z
        table[f"{name}_q"] = q
        matched = np.full(len(pw), np.nan)
        near_idx = {}
        for i in range(len(pw)):
            cand = np.where(~kind & (np.arange(len(pw)) != i))[0]
            near_idx[i] = cand[np.argsort(((feat[cand] - feat[i]) ** 2).sum(1))[:50]]
            matched[i] = (T[near_idx[i]] < T[i]).mean()
        table[f"{name}_matched_pct"] = matched
        # concept-level test against 2000 random sets of frequency-matched non-concept words
        # (each form is swapped for a random one of its 50 nearest words in mean frequency and
        # frequency ratio); this is relative to the global paper-level offset, and parallels the
        # pseudo-concept test used for salience.
        for (grp, cn), idx in table[table.group.notna()].groupby(["group", "concept"]).groups.items():
            forms = list(idx)
            sims = np.mean([T[rs.choice(near_idx[f], 2000)] for f in forms], axis=0)
            obs = T[forms].mean()
            concept_rows.append({"measure": name, "group": grp, "concept": cn, "forms": len(forms), "shift": obs,
                                 "matched_set_mean": sims.mean(), "matched_set_sd": sims.std(),
                                 "percentile": float((sims < obs).mean()),
                                 "p_greater": float((1 + (sims >= obs).sum()) / 2001)})
        summary[f"global_{name}"] = g
        summary[f"n_sig_q05_{name}"] = int((q < 0.05).sum())
        summary[f"spearman_{name}_vs_abslogratio"] = float(pd.Series(T).corr(pd.Series(np.abs(table.log2_ratio)), method="spearman"))
    # directional (signed, two-sided)
    r_, n_ = stat["delta"]
    Td = r_[:, probe].mean(0)
    avg = averaged_null(n_[:, probe], D, rs)
    mu, sd = avg.mean(0), np.maximum(avg.std(0, ddof=1), 1e-9)
    table["delta_isig_minus_imvo"], table["delta_z"] = Td, (Td - mu) / sd
    # concept-level directional test: mean Delta over forms of each concept vs resampled null
    rows = []
    for (grp, cn), idx in table[table.group.notna()].groupby(["group", "concept"]).groups.items():
        forms = [i for i in idx if not any(s in table.word[i] for s in SEED_STEMS)]
        if not forms:
            continue
        obs = Td[forms].mean()
        nul = avg[:, forms].mean(1)
        rows.append({"group": grp, "concept": cn, "forms": len(forms), "delta_isig_minus_imvo": obs,
                     "null_mean": nul.mean(), "null_sd": nul.std(ddof=1),
                     "z": (obs - nul.mean()) / nul.std(ddof=1),
                     "p_two_sided": float((1 + (np.abs(nul - nul.mean()) >= abs(obs - nul.mean())).sum()) / (NRESAMPLE + 1))})
    dir_tab = pd.DataFrame(rows)
    allt = [i for i in range(len(pw)) if table.group[i] == "theory" and not any(s in pw[i] for s in SEED_STEMS)]
    nul = avg[:, allt].mean(1); obs = Td[allt].mean()
    summary["directional_all_theory"] = {"forms": len(allt), "delta": float(obs), "null_mean": float(nul.mean()),
                                         "null_sd": float(nul.std(ddof=1)), "z": float((obs - nul.mean()) / nul.std(ddof=1)),
                                         "p_two_sided": float((1 + (np.abs(nul - nul.mean()) >= abs(obs - nul.mean())).sum()) / (NRESAMPLE + 1))}
    summary["directional_all_words_mean_delta"] = float(Td.mean())
    table.to_csv(OUT / f"framing_words{tag}.csv", index=False)
    pd.DataFrame(concept_rows).to_csv(OUT / f"framing_concepts{tag}.csv", index=False)
    dir_tab.to_csv(OUT / f"directional_by_concept{tag}.csv", index=False)

    # stable neighbours (for the speaker sheet and for abantu): share of real models listing each neighbour
    disp = sorted({w for w in pw if concept(w)[0] is not None} | {"abantu", "umntu", "abantsundu", "abamnyama"} & set(words))
    nbr = []
    for w in disp:
        i = words.index(w)
        for side, key in (("Isigidimi", "ri"), ("Imvo", "rm")):
            c = {}
            for d in range(D):
                for j in at[(key, d)]["nb"][i]:
                    c[words[j]] = c.get(words[j], 0) + 1
            for nb_w, cnt in sorted(c.items(), key=lambda kv: -kv[1])[:12]:
                nbr.append({"word": w, "paper": side, "neighbour": nb_w, "in_models": cnt, "of": D})
    pd.DataFrame(nbr).to_csv(OUT / f"neighbours_stable{tag}.csv", index=False)

    summary["variant_neighbour_share_isig"] = variant_share(words, at[("ri", 0)]["nb"], probe)
    summary["variant_neighbour_share_imvo"] = variant_share(words, at[("rm", 0)]["nb"], probe)
    summary["real_by_draw"] = {n: [float(r[probe].mean()) for r in stat[n][0]] for n in ("profile", "jaccard")}
    (OUT / f"framing_summary{tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    np.savez_compressed(OUT / f"framing_raw{tag}.npz", words=np.array(words), probe=probe,
                        **{f"real_{k}": v[0] for k, v in stat.items()}, **{f"null_{k}": v[1] for k, v in stat.items()})
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("anchors")}, indent=2))
    print(pd.DataFrame(concept_rows).round(3).to_string(index=False))
    print(dir_tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
