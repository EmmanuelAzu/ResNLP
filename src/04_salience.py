"""Salience: how often does each paper use a concept stem (per million tokens)?

For every comparison the concept's log2(rate ratio) is judged two ways:
  (1) issue-level bootstrap CI of the ratio (4000 resamples of issues within each period);
  (2) against 2000 pseudo-concepts: random bags of word types, excluded from all concept lists,
      whose combined count matches the concept's (+-15%) - the same frequency-matching idea as
      the framing test.  p = share of pseudo-concepts with |log2 ratio| >= the concept's.
Comparisons
  main      Isigidimi vs Imvo, same year-months, Nov 1884-1888
  heldout   Isigidimi 1876-83 vs Imvo 1889-94 (replication window; time confounded)
  jabavu    rates by period: Isigidimi 1876-80 | Isigidimi 1881-83 (Jabavu's editorship, provisional) |
            Isigidimi Nov 1884-88 | Imvo Nov 1884-86
"""
import numpy as np
import pandas as pd

from common import CONTROL, EXPLORATORY, OUT, THEORY, load_pages

B, NPSEUDO = 4000, 2000
rng = np.random.default_rng(0)
STEMS = {f"{g}:{n}": s for g, t in (("theory", THEORY), ("control", CONTROL), ("exploratory", EXPLORATORY))
         for n, s in t.items()}
ALL_STEMS = [s for v in STEMS.values() for s in v]
pages = load_pages(window=None)
pages["year"] = pages.date.str[:4].astype(int)


def select(paper, lo, hi):
    return pages[(pages.paper == paper) & (pages.date >= lo) & (pages.date <= hi)]


def per_issue(df):
    rows, types = [], {}
    for date, g in df.groupby("date"):
        toks = [t for ss in g.sents for s in ss for t in s]
        row = {"tokens": len(toks), **{k: sum(any(st in t for st in ss) for t in toks) for k, ss in STEMS.items()}}
        rows.append(row)
        for t in toks:
            types[t] = types.get(t, 0) + 1
    return pd.DataFrame(rows), types


def rate(t, idx, key):
    return 1e6 * t.iloc[idx][key].sum() / t.iloc[idx].tokens.sum()


def boot_rates(t, key):
    return np.array([rate(t, rng.integers(0, len(t), len(t)), key) for _ in range(B)])


def pseudo_concepts(ta, tb, na, nb, target, n=NPSEUDO):
    """log2 ratios of random bags of types whose total count matches `target`."""
    cand = [w for w in set(ta) | set(tb) if ta.get(w, 0) + tb.get(w, 0) >= 10 and not any(s in w for s in ALL_STEMS)]
    cnt = np.array([ta.get(w, 0) + tb.get(w, 0) for w in cand])
    ca = np.array([ta.get(w, 0) for w in cand]); cb = np.array([tb.get(w, 0) for w in cand])
    out = []
    for _ in range(n):
        order = rng.permutation(len(cand)); cs = np.cumsum(cnt[order])
        k = np.searchsorted(cs, target * rng.uniform(0.85, 1.15)) + 1
        sel = order[:k]
        out.append(np.log2(((cb[sel].sum() + .5) / nb) / ((ca[sel].sum() + .5) / na)))
    return np.array(out)


def compare(label, a_df, b_df, names=("Isigidimi", "Imvo")):
    ta, tya = per_issue(a_df); tb, tyb = per_issue(b_df)
    na, nb = sum(tya.values()), sum(tyb.values())
    rows = []
    for key in STEMS:
        ra, rb = rate(ta, np.arange(len(ta)), key), rate(tb, np.arange(len(tb)), key)
        ba, bb = boot_rates(ta, key), boot_rates(tb, key)
        ratio = (bb + 1e-9) / (ba + 1e-9)
        pseudo = pseudo_concepts(tya, tyb, na, nb, ta[key].sum() + tb[key].sum())
        l2 = np.log2((rb + 1e-9) / (ra + 1e-9))
        rows.append({"comparison": label, "concept": key, f"per_M_{names[0]}": ra, f"per_M_{names[1]}": rb,
                     "ratio": rb / ra if ra else np.nan, "ci_lo": np.percentile(ratio, 2.5), "ci_hi": np.percentile(ratio, 97.5),
                     "ci_excludes_1": bool(np.percentile(ratio, 2.5) > 1 or np.percentile(ratio, 97.5) < 1),
                     "pseudo_p": float((1 + (np.abs(pseudo) >= abs(l2)).sum()) / (len(pseudo) + 1)),
                     "pseudo_sd_log2": float(pseudo.std())})
    print(f"\n{label}: {len(ta)}+{len(tb)} issues, {na:,}+{nb:,} tokens")
    df = pd.DataFrame(rows)
    print(df.drop(columns="comparison").round(3).to_string(index=False))
    return df


# ---- main: same year-months --------------------------------------------------------------
isig_main = select("ISIG", "1884-11-01", "1888-12-31")
imvo_main = select("IMVO", "1884-11-01", "1888-12-31")
imvo_main = imvo_main[imvo_main.date.str[:7].isin(set(isig_main.date.str[:7]))]
main = compare("main", isig_main, imvo_main)
main.to_csv(OUT / "salience.csv", index=False)

# ---- replication on held-out windows ---------------------------------------------------------
held = compare("heldout", select("ISIG", "1876-01-01", "1883-12-31"), select("IMVO", "1889-01-01", "1894-12-31"))
held.to_csv(OUT / "salience_heldout.csv", index=False)

# ---- Jabavu contrast: rates by period ------------------------------------------------------
periods = {"Isigidimi 1876-80": select("ISIG", "1876-01-01", "1880-12-31"),
           "Isigidimi 1881-83 (Jabavu)": select("ISIG", "1881-01-01", "1883-12-31"),
           "Isigidimi Nov84-88": isig_main,
           "Imvo Nov84-86": select("IMVO", "1884-11-01", "1886-12-31")}
rows = []
tabs = {k: per_issue(v)[0] for k, v in periods.items()}
for key in ("exploratory:religion", "theory:parliament", "theory:vote", "theory:rule", "theory:government",
            "control:time", "control:house"):
    for k, t in tabs.items():
        bt = boot_rates(t, key)
        rows.append({"concept": key, "period": k, "issues": len(t), "tokens": int(t.tokens.sum()),
                     "per_M": rate(t, np.arange(len(t)), key), "ci_lo": np.percentile(bt, 2.5), "ci_hi": np.percentile(bt, 97.5)})
jab = pd.DataFrame(rows)
jab.to_csv(OUT / "salience_jabavu.csv", index=False)
print("\njabavu contrast (per million tokens)")
print(jab.pivot(index="concept", columns="period", values="per_M").round(0).to_string())
