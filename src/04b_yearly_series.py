"""Yearly rates (per million tokens) of religious and parliamentary vocabulary in each paper,
with issue-level bootstrap CIs.  Used for the Jabavu-contrast figure (his Isigidimi years are
provisional: see README)."""
import numpy as np
import pandas as pd

from common import EXPLORATORY, OUT, THEORY, load_pages

B = 1000
rng = np.random.default_rng(3)
CONCEPTS = {"religion": EXPLORATORY["religion"], "parliament": THEORY["parliament"], "vote": THEORY["vote"]}
pages = load_pages(window=None)
pages["year"] = pages.date.str[:4].astype(int)
rows = []
for (paper, year), g in pages.groupby(["paper", "year"]):
    issues = []
    for _, gi in g.groupby("date"):
        toks = [t for ss in gi.sents for s in ss for t in s]
        issues.append({"tokens": len(toks), **{k: sum(any(s in t for s in st) for t in toks) for k, st in CONCEPTS.items()}})
    t = pd.DataFrame(issues)
    for k in CONCEPTS:
        pt = 1e6 * t[k].sum() / t.tokens.sum()
        bs = [1e6 * t.iloc[i][k].sum() / t.iloc[i].tokens.sum() for i in (rng.integers(0, len(t), len(t)) for _ in range(B))]
        rows.append({"paper": paper, "year": year, "concept": k, "issues": len(t), "tokens": int(t.tokens.sum()),
                     "per_M": pt, "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
df = pd.DataFrame(rows)
df.to_csv(OUT / "salience_yearly.csv", index=False)
print(df[df.concept == "religion"].pivot(index="year", columns="paper", values="per_M").round(0).to_string())
