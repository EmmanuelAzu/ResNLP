"""Recompute concept membership and the concept-level matched-set framing tests from an existing
per-word table (framing_words{TAG}.csv), WITHOUT retraining any model.

Use when the stem definitions in common.py change (e.g. the 'law' stem 'nteto' was dropped because it
also matches 'intetho', speech).  The per-word shifts are unchanged; only which words count as
concept words (and hence which words are eligible as frequency-matched controls) changes.

  python src/11_reassign_concepts.py _w5          (files in outputs/)
  FRAMING_DIR=outputs/window2 python src/11_reassign_concepts.py _segnorm
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import CONTROL, OUT, THEORY

D = Path(os.environ.get("FRAMING_DIR", OUT))
TAG = sys.argv[1] if len(sys.argv) > 1 else ""


def concept(word):
    for group, table in (("theory", THEORY), ("control", CONTROL)):
        for name, stems in table.items():
            if any(s in word for s in stems):
                return group, name
    return None, None


t = pd.read_csv(D / f"framing_words{TAG}.csv")
gc = t.word.map(concept)
t["group"], t["concept"] = [g for g, _ in gc], [c for _, c in gc]
fi, fm = t.n_isig.values, t.n_imvo_draw.values
feat = np.c_[np.log10((fi + fm) / 2), np.abs(np.log2(fm / fi))]
feat = (feat - feat.mean(0)) / feat.std(0)
kind = t.group.notna().values
rs = np.random.default_rng(11)
rows = []
for name in ("profile", "jaccard"):
    T = t[f"{name}_shift"].values
    near, matched = {}, np.full(len(t), np.nan)
    for i in range(len(t)):
        cand = np.where(~kind & (np.arange(len(t)) != i))[0]
        near[i] = cand[np.argsort(((feat[cand] - feat[i]) ** 2).sum(1))[:50]]
        matched[i] = (T[near[i]] < T[i]).mean()
    t[f"{name}_matched_pct"] = matched
    for (grp, cn), idx in t[t.group.notna()].groupby(["group", "concept"]).groups.items():
        forms = list(idx)
        sims = np.mean([T[rs.choice(near[f], 2000)] for f in forms], axis=0)
        obs = T[forms].mean()
        rows.append({"measure": name, "group": grp, "concept": cn, "forms": len(forms), "shift": obs,
                     "matched_set_mean": sims.mean(), "matched_set_sd": sims.std(),
                     "percentile": float((sims < obs).mean()), "p_greater": float((1 + (sims >= obs).sum()) / 2001)})
t.to_csv(D / f"framing_words{TAG}.csv", index=False)
pd.DataFrame(rows).to_csv(D / f"framing_concepts{TAG}.csv", index=False)
c = pd.DataFrame(rows)
print(D.name, TAG or "(base)", "law forms:", int(c[(c.measure == "profile") & (c.concept == "law")].forms.iloc[0]))
