"""How much does a concept's rate move within ONE paper across eras?  Calibrates the held-out
comparison (Isigidimi 1876-83 vs Imvo 1889-94), which mixes paper and era.
Reports per-million rates and within-paper era ratios (issue-bootstrap CIs)."""
import numpy as np
import pandas as pd

from common import EXPLORATORY, OUT, THEORY, load_pages

B = 2000
rng = np.random.default_rng(4)
STEMS = {"parliament": THEORY["parliament"], "government": THEORY["government"], "vote": THEORY["vote"],
         "religion": EXPLORATORY["religion"]}
pages = load_pages(window=None)


def issues(paper, lo, hi):
    g = pages[(pages.paper == paper) & (pages.date >= lo) & (pages.date <= hi)]
    rows = []
    for _, gi in g.groupby("date"):
        toks = [t for ss in gi.sents for s in ss for t in s]
        rows.append({"tokens": len(toks), **{k: sum(any(s in t for s in st) for t in toks) for k, st in STEMS.items()}})
    return pd.DataFrame(rows)


def rate(t, idx, k):
    return 1e6 * t.iloc[idx][k].sum() / t.iloc[idx].tokens.sum()


def boot(t, k):
    return np.array([rate(t, rng.integers(0, len(t), len(t)), k) for _ in range(B)])


periods = {"Isigidimi 1876-83": issues("ISIG", "1876-01-01", "1883-12-31"),
           "Isigidimi Nov84-88": issues("ISIG", "1884-11-01", "1888-12-31"),
           "Imvo Nov84-88": issues("IMVO", "1884-11-01", "1888-12-31"),
           "Imvo 1889-94": issues("IMVO", "1889-01-01", "1894-12-31")}
pairs = [("Isigidimi 1876-83", "Isigidimi Nov84-88", "Isigidimi: later / earlier era"),
         ("Imvo Nov84-88", "Imvo 1889-94", "Imvo: later / earlier era"),
         ("Isigidimi Nov84-88", "Imvo Nov84-88", "Imvo / Isigidimi, same era (main)"),
         ("Isigidimi 1876-83", "Imvo 1889-94", "Imvo 1889-94 / Isigidimi 1876-83 (held-out)")]
rows = []
for k in STEMS:
    for a, b, label in pairs:
        ba, bb = boot(periods[a], k), boot(periods[b], k)
        r = (bb + 1e-9) / (ba + 1e-9)
        rows.append({"concept": k, "comparison": label, "per_M_first": rate(periods[a], np.arange(len(periods[a])), k),
                     "per_M_second": rate(periods[b], np.arange(len(periods[b])), k),
                     "ratio": rate(periods[b], np.arange(len(periods[b])), k) / max(rate(periods[a], np.arange(len(periods[a])), k), 1e-9),
                     "ci_lo": np.percentile(r, 2.5), "ci_hi": np.percentile(r, 97.5)})
df = pd.DataFrame(rows)
df.to_csv(OUT / "era_drift.csv", index=False)
print(df.round(2).to_string(index=False))
