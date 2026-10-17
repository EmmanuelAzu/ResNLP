"""Salience: how often does each paper use a concept stem (per million tokens)?

All Isigidimi issues in the window vs ALL Imvo issues from the same year-months (rates do
not need equal sizes, so no subsampling). Uncertainty: issue-level bootstrap (resample
issues with replacement within each paper), 4000 replicates; ratio = Imvo / Isigidimi.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import CONTROL, OUT, THEORY, load_pages

B = 4000
rng = np.random.default_rng(0)
pages = load_pages()
months = set(pages[pages.paper == "ISIG"].ym)
pages = pages[pages.ym.isin(months)]

stems = {f"{g}:{n}": s for g, t in (("theory", THEORY), ("control", CONTROL)) for n, s in t.items()}
per_issue = {}
for (paper, date), g in pages.groupby(["paper", "date"]):
    toks = [t for ss in g.sents for s in ss for t in s]
    row = {"tokens": len(toks)}
    for key, ss in stems.items():
        row[key] = sum(any(st in t for st in ss) for t in toks)
    per_issue.setdefault(paper, []).append(row)
tabs = {p: pd.DataFrame(v) for p, v in per_issue.items()}
print({p: (len(t), int(t.tokens.sum())) for p, t in tabs.items()})


def rate(t, idx):
    return 1e6 * t.iloc[idx].drop(columns="tokens").sum() / t.iloc[idx].tokens.sum()


out = []
boot = {p: [] for p in tabs}
for p, t in tabs.items():
    for _ in range(B):
        boot[p].append(rate(t, rng.integers(0, len(t), len(t))))
for key in stems:
    ri = rate(tabs["ISIG"], np.arange(len(tabs["ISIG"])))[key]
    rm = rate(tabs["IMVO"], np.arange(len(tabs["IMVO"])))[key]
    bi = np.array([b[key] for b in boot["ISIG"]]); bm = np.array([b[key] for b in boot["IMVO"]])
    ratio = (bm + 1e-9) / (bi + 1e-9)
    lo, hi = np.percentile(ratio, [2.5, 97.5])
    out.append({"concept": key, "per_M_isig": ri, "per_M_imvo": rm, "ratio": rm / ri,
                "ci_lo": lo, "ci_hi": hi, "ci_excludes_1": bool(lo > 1 or hi < 1)})
df = pd.DataFrame(out).sort_values("ratio", ascending=False)
df.to_csv(OUT / "salience.csv", index=False)
print(df.round(2).to_string(index=False))

# forest plot of the ratio on a log axis
fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
ink, muted = "#0b0b0b", "#52514e"
d = df.iloc[::-1].reset_index(drop=True)
for i, r in d.iterrows():
    th = r.concept.startswith("theory")
    ax.plot([r.ci_lo, r.ci_hi], [i, i], color=muted, lw=1)
    ax.plot(r.ratio, i, "o", ms=5, mfc=ink if th else "white", mec=ink, mew=1)
ax.axvline(1, color=muted, lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_xticks([0.25, 0.5, 1, 2, 4])
ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
ax.minorticks_off()
ax.set_yticks(range(len(d)))
ax.set_yticklabels([c.split(":")[1] for c in d.concept], fontsize=7, color=ink)
ax.set_xlabel("Imvo / Isigidimi rate per million tokens (95% issue-bootstrap CI)", fontsize=7, color=ink)
ax.tick_params(axis="x", labelsize=7)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.plot([], [], "o", mfc=ink, mec=ink, label="political concept")
ax.plot([], [], "o", mfc="white", mec=ink, label="neutral control")
ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig_salience.png")
print("saved fig_salience.png")


# ---- exploratory: religious vocabulary (suggested by the classifier's top features, so the CI is
# optimistic: it was chosen BECAUSE it separates the papers). Not part of the pre-specified list.
EXPLORATORY = {"religion": ["tixo", "kristu"]}
ex = {}
for paper, t in per_issue.items():
    pass
rows = []
for paper, g in pages.groupby("paper"):
    for date, gi in g.groupby("date"):
        toks = [t for ss in gi.sents for s in ss for t in s]
        rows.append({"paper": paper, "tokens": len(toks),
                     **{k: sum(any(st in t for st in v) for t in toks) for k, v in EXPLORATORY.items()}})
ex = pd.DataFrame(rows)
out2 = []
for k in EXPLORATORY:
    r = {}
    for paper, g in ex.groupby("paper"):
        reps = [1e6 * g.iloc[i][k].sum() / g.iloc[i].tokens.sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(B))]
        r[paper] = (1e6 * g[k].sum() / g.tokens.sum(), reps)
    ratio = np.array(r["IMVO"][1]) / (np.array(r["ISIG"][1]) + 1e-9)
    out2.append({"concept": k, "per_M_isig": r["ISIG"][0], "per_M_imvo": r["IMVO"][0],
                 "ratio": r["IMVO"][0] / r["ISIG"][0], "ci_lo": np.percentile(ratio, 2.5), "ci_hi": np.percentile(ratio, 97.5)})
pd.DataFrame(out2).to_csv(OUT / "salience_exploratory.csv", index=False)
print("exploratory:"); print(pd.DataFrame(out2).round(2).to_string(index=False))
