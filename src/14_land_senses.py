"""Which sense of umhlaba ('land' / 'earth, world') does each paper use?

For every token containing the land stem (mhlaba), look at the 10 tokens either side and record
whether a religious cue (heaven, God, Christ, Saviour, holy) or a property/political cue (title
deed, law, government, parliament, magistrate, farm, vote, tax, location) occurs.  Imvo is
restricted to the year-months in which Isigidimi has text.  Writes outputs/land_senses.csv and
outputs/land_kwic.md (random example lines per paper and sense, for a speaker to check).
"""
import random

import numpy as np
import pandas as pd

from common import OUT, THEORY, load_pages

STEM = THEORY["land"]
RELIGIOUS = ("zulwini", "tixo", "kristu", "yesu", "msindisi", "ngcwele", "tixweni")
RELIGIOUS_EXACT = {"izulu", "lizulu", "nezulu"}  # not bare "zulu": kwa zulu = Zululand
POLITICAL = ("taitile", "mteto", "rulumente", "palamente", "mantyi", "fama", "voti", "nyulo", "rafu", "lokishi")
W = 10
rng = np.random.default_rng(3)
random.seed(3)


def cue(tokens):
    rel = any(t in RELIGIOUS_EXACT or any(c in t for c in RELIGIOUS) for t in tokens)
    pol = any(any(c in t for c in POLITICAL) for t in tokens)
    return rel, pol


pages = load_pages()
pages = pages[pages.ym.isin(set(pages[pages.paper == "ISIG"].ym))]
rows, kwic = [], []
for _, r in pages.iterrows():
    toks = [t for s in r.sents for t in s]
    for i, t in enumerate(toks):
        if any(s in t for s in STEM):
            ctx = toks[max(0, i - W):i] + toks[i + 1:i + 1 + W]
            rel, pol = cue(ctx)
            rows.append({"paper": r.paper, "date": r.date, "form": t, "religious_cue": rel, "political_cue": pol})
            kwic.append((r.paper, rel, pol, " ".join(toks[max(0, i - W):i]) + f" [{t}] " + " ".join(toks[i + 1:i + 1 + W])))
df = pd.DataFrame(rows)
iss = df.groupby(["paper", "date"]).agg(n=("form", "size"), rel=("religious_cue", "sum"), pol=("political_cue", "sum")).reset_index()
out = []
for p, g in iss.groupby("paper"):
    g = g.reset_index(drop=True)
    for k in ("rel", "pol"):
        share = g[k].sum() / g.n.sum()
        boot = [g.iloc[idx][k].sum() / g.iloc[idx].n.sum() for idx in (rng.integers(0, len(g), len(g)) for _ in range(4000))]
        out.append({"paper": p, "cue": {"rel": "religious", "pol": "property/political"}[k], "uses": int(g.n.sum()),
                    "share": share, "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5)})
res = pd.DataFrame(out)
res.to_csv(OUT / "land_senses.csv", index=False)
print(res.round(3).to_string())
with open(OUT / "land_kwic.md", "w") as fh:
    fh.write("# umhlaba in context (10 tokens either side); random examples\n\n")
    for p in ("ISIG", "IMVO"):
        for lab, sel in (("religious cue", lambda x: x[1] and not x[2]), ("property/political cue", lambda x: x[2] and not x[1]),
                         ("no cue", lambda x: not x[1] and not x[2])):
            pool = [k for k in kwic if k[0] == p and sel(k)]
            fh.write(f"## {p}: {lab} ({len(pool)} uses)\n\n")
            for k in random.sample(pool, min(5, len(pool))):
                fh.write(f"- {k[3]}\n")
            fh.write("\n")
