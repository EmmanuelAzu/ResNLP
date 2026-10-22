"""Hold the editor constant, vary control (and vice versa).

Three cells (editors after Schoots 2023b, p. 71):
  J-mission : Isigidimi, Jan 1881 - Oct 1884   (Jabavu edits a mission paper)
  J-free    : Imvo,      Nov 1884 - Dec 1888   (Jabavu edits an African-owned paper)
  G-mission : Isigidimi, Nov 1884 - Dec 1888   (Gqoba edits the mission paper)
J-free vs J-mission  = same editor, different control (confounded with time);
G-mission vs J-mission = same control, different editor (also confounded with time).
Time is calibrated by Isigidimi's own drift between 1876-80 and 1881-84 under its earlier editor.
Rates per million tokens; 95% CIs of ratios from 4000 issue resamples.  Writes outputs/editor_control.csv.
"""
import numpy as np
import pandas as pd

from common import EXPLORATORY, OUT, THEORY, load_pages

CELLS = {"J-mission": ("ISIG", "1881-01-01", "1884-10-31"), "J-free": ("IMVO", "1884-11-01", "1888-12-31"),
         "G-mission": ("ISIG", "1884-11-01", "1888-12-31"), "early-mission": ("ISIG", "1876-01-01", "1880-12-31")}
CONCEPTS = {**{k: v for k, v in THEORY.items()}, "religion": EXPLORATORY["religion"]}
PAIRS = [("J-free", "J-mission", "control (same editor)"), ("G-mission", "J-mission", "editor (same control)"),
         ("J-mission", "early-mission", "time within Isigidimi (1881-84 vs 1876-80)")]
rng = np.random.default_rng(21)
pages = load_pages(window=None)
iss = {}
for cell, (paper, lo, hi) in CELLS.items():
    df = pages[(pages.paper == paper) & (pages.date >= lo) & (pages.date <= hi)]
    rows = []
    for d, g in df.groupby("date"):
        toks = [t for ss in g.sents for s in ss for t in s]
        rows.append({"tokens": len(toks), **{c: sum(any(x in t for x in st) for t in toks) for c, st in CONCEPTS.items()}})
    iss[cell] = pd.DataFrame(rows)
rate = lambda df, c: 1e6 * df[c].sum() / df.tokens.sum()
out = []
for a, b, lab in PAIRS:
    A, B = iss[a], iss[b]
    for c in CONCEPTS:
        ra, rb = rate(A, c), rate(B, c)
        boot = []
        for _ in range(4000):
            x = rate(A.iloc[rng.integers(0, len(A), len(A))], c); y = rate(B.iloc[rng.integers(0, len(B), len(B))], c)
            boot.append(x / y if y > 0 else np.nan)
        out.append({"contrast": lab, "num": a, "den": b, "concept": c, "per_M_num": ra, "per_M_den": rb,
                    "tokens_num": int(A.tokens.sum()), "tokens_den": int(B.tokens.sum()), "issues_num": len(A), "issues_den": len(B),
                    "ratio": ra / rb if rb else np.nan, "ci_lo": np.nanpercentile(boot, 2.5), "ci_hi": np.nanpercentile(boot, 97.5)})
res = pd.DataFrame(out)
res.to_csv(OUT / "editor_control.csv", index=False)
pd.set_option("display.width", 220)
print(res[res.concept.isin(["religion", "parliament", "vote", "government", "law", "land", "rights"])][
    ["contrast", "concept", "per_M_num", "per_M_den", "ratio", "ci_lo", "ci_hi", "issues_num", "issues_den"]].round(2).to_string())
