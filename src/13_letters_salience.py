"""Where does the agenda gap live: in readers' letters or in the rest of the paper?

Splits every page with 12_letters.split_page, then computes concept rates (per million tokens)
for letters and for everything else, in each paper.  Imvo is restricted to the year-months in
which Isigidimi has text (date matching, as in the main analysis).  95% CIs of the Imvo/Isigidimi
rate ratio come from 4000 bootstrap resamples of issues.  Writes outputs/letters_salience.csv.
"""
import importlib

import numpy as np
import pandas as pd

from common import EXPLORATORY, OUT, THEORY, load_pages

split_page = importlib.import_module("12_letters").split_page
CONCEPTS = {"parliament": THEORY["parliament"], "religion": EXPLORATORY["religion"],
            "vote": THEORY["vote"], "government": THEORY["government"]}
B = 4000
rng = np.random.default_rng(11)

pages = load_pages()
pages = pages[pages.ym.isin(set(pages[pages.paper == "ISIG"].ym))]
rows = []
for _, r in pages.iterrows():
    flag = split_page(r.sents)
    for part in ("letters", "rest"):
        toks = [t for s, f in zip(r.sents, flag) if f == (part == "letters") for t in s]
        rows.append({"paper": r.paper, "date": r.date, "part": part, "tokens": len(toks),
                     **{c: sum(any(st in t for st in ss) for t in toks) for c, ss in CONCEPTS.items()}})
iss = pd.DataFrame(rows).groupby(["paper", "date", "part"]).sum(numeric_only=True).reset_index()


def rate(df, c):
    return 1e6 * df[c].sum() / df.tokens.sum()


out = []
for part in ("letters", "rest"):
    a = iss[(iss.paper == "ISIG") & (iss.part == part)].reset_index(drop=True)
    b = iss[(iss.paper == "IMVO") & (iss.part == part)].reset_index(drop=True)
    for c in CONCEPTS:
        ra, rb = rate(a, c), rate(b, c)
        boot = []
        for _ in range(B):
            sa, sb = a.iloc[rng.integers(0, len(a), len(a))], b.iloc[rng.integers(0, len(b), len(b))]
            x, y = rate(sa, c), rate(sb, c)
            boot.append(y / x if x > 0 else np.nan)
        boot = np.array(boot)
        out.append({"part": part, "concept": c, "tokens_isig": int(a.tokens.sum()), "tokens_imvo": int(b.tokens.sum()),
                    "hits_isig": int(a[c].sum()), "hits_imvo": int(b[c].sum()),
                    "per_M_isig": ra, "per_M_imvo": rb, "ratio": rb / ra if ra else np.nan,
                    "ci_lo": np.nanpercentile(boot, 2.5), "ci_hi": np.nanpercentile(boot, 97.5),
                    "share_nan_boot": float(np.isnan(boot).mean())})
res = pd.DataFrame(out)
res.to_csv(OUT / "letters_salience.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string())
