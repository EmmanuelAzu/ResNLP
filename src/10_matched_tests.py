"""Calibrate framing statistics against frequency-matched word sets (reads outputs/framing_words.csv).

  1. pooled electoral-institutional set (palamente/voti/rulumente forms) on the profile shift
  2. directional Delta (Isigidimi minus Imvo; + = Isigidimi closer to religious than political anchors),
     per concept and pooled, vs matched sets (two-sided).  Seed words (tixo, kristu, palamente, voti)
     are excluded from targets AND from the matched pool.
  3. data-driven list: words whose profile shift most exceeds that of their 50 frequency-matched words
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT

D = Path(os.environ.get("FRAMING_DIR", OUT))

SEED = ("tixo", "kristu", "palamente", "voti")
rng = np.random.default_rng(5)
import sys
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
t = pd.read_csv(D / f"framing_words{TAG}.csv")
feat = np.c_[np.log10((t.n_isig + t.n_imvo_draw) / 2), np.abs(t.log2_ratio)]
feat = (feat - feat.mean(0)) / feat.std(0)
is_concept = t.group.notna().values
is_seed = t.word.map(lambda w: any(s in w for s in SEED)).values
pool = ~is_concept & ~is_seed
near = {}
for i in range(len(t)):
    cand = np.where(pool & (np.arange(len(t)) != i))[0]
    near[i] = cand[np.argsort(((feat[cand] - feat[i]) ** 2).sum(1))[:50]]


def set_test(forms, col, two_sided=False, n=2000):
    v = t[col].values
    sims = np.mean([v[rng.choice(near[f], n)] for f in forms], axis=0)
    obs = v[forms].mean()
    if two_sided:
        p = (1 + (np.abs(sims - sims.mean()) >= abs(obs - sims.mean())).sum()) / (n + 1)
    else:
        p = (1 + (sims >= obs).sum()) / (n + 1)
    return obs, sims.mean(), sims.std(), float((sims < obs).mean()), float(p)


rows = []
POOLS = {"palamente+voti (parliament, vote)": ("parliament", "vote"),
         "palamente+voti+rulumente (adds government)": ("parliament", "vote", "government")}
for label, concepts in POOLS.items():
    forms_ = [i for i in range(len(t)) if t.concept[i] in concepts]
    for col, mname in (("profile_shift", "profile"), ("jaccard_shift", "Jaccard")):
        o = set_test(forms_, col)
        rows.append({"test": f"{mname} shift: {label}", "forms": len(forms_), "obs": o[0], "matched_mean": o[1],
                     "matched_sd": o[2], "percentile": o[3], "p": o[4],
                     "detectable_excess_95": 1.645 * o[2]})
for (grp, cn), idx in t[t.group.notna()].groupby(["group", "concept"]).groups.items():
    forms = [i for i in idx if not is_seed[i]]
    if forms:
        o = set_test(forms, "delta_isig_minus_imvo", two_sided=True)
        rows.append({"test": f"directional delta: {grp}:{cn}", "forms": len(forms), "obs": o[0], "matched_mean": o[1],
                     "matched_sd": o[2], "percentile": o[3], "p": o[4]})
for label, sel in (("all political concepts", lambda g: g == "theory"), ("all neutral controls", lambda g: g == "control")):
    forms = [i for i in range(len(t)) if sel(t.group[i]) and not is_seed[i]]
    o = set_test(forms, "delta_isig_minus_imvo", two_sided=True)
    rows.append({"test": f"directional delta: {label}", "forms": len(forms), "obs": o[0], "matched_mean": o[1],
                 "matched_sd": o[2], "percentile": o[3], "p": o[4]})
# leave-one-concept-out sensitivity of the pooled directional test (political concepts)
all_forms = [i for i in range(len(t)) if t.group[i] == "theory" and not is_seed[i]]
for cn in sorted({t.concept[i] for i in all_forms}):
    forms = [i for i in all_forms if t.concept[i] != cn]
    o = set_test(forms, "delta_isig_minus_imvo", two_sided=True)
    rows.append({"test": f"directional delta: political concepts without {cn}", "forms": len(forms), "obs": o[0], "matched_mean": o[1],
                 "matched_sd": o[2], "percentile": o[3], "p": o[4]})
res = pd.DataFrame(rows)
res.to_csv(D / f"framing_matched_tests{TAG}.csv", index=False)
print(res.round(3).to_string(index=False))

# data-driven list
v = t.profile_shift.values
zm = np.array([(v[i] - v[near[i]].mean()) / v[near[i]].std() for i in range(len(t))])
t["z_vs_matched"] = zm
top = t[~is_seed].sort_values("z_vs_matched", ascending=False).head(25)
top[["word", "n_isig", "n_imvo_draw", "log2_ratio", "profile_shift", "z_vs_matched", "jaccard_shift"]].round(3).to_csv(D / f"most_shifted_words{TAG}.csv", index=False)
print("\nmost shifted vs matched words (profile):")
print(top[["word", "n_isig", "n_imvo_draw", "log2_ratio", "profile_shift", "z_vs_matched"]].round(2).to_string(index=False))
