"""Figures for the write-up (salience + framing per concept, global null, fertility)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # validated categorical slots 1-3
plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False})


def two_part():
    sal = pd.read_csv(OUT / "salience.csv")
    fw = pd.read_csv(OUT / "framing_words.csv")
    sal["name"] = sal.concept.str.split(":").str[1]
    sal["theory"] = sal.concept.str.startswith("theory")
    order = sal.sort_values("ratio")["name"].tolist()          # bottom -> top
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 3.0), dpi=200, sharey=True,
                               gridspec_kw={"wspace": 0.08})
    for i, n in enumerate(order):
        r = sal[sal.name == n].iloc[0]
        a.plot([r.ci_lo, r.ci_hi], [i, i], color=MUTED, lw=1)
        a.plot(r.ratio, i, "o", ms=5, mfc=INK if r.theory else "white", mec=INK, mew=1)
        f = fw[fw.concept == n]
        if len(f):
            b.plot(f.matched_percentile, [i + (k - (len(f) - 1) / 2) * 0.07 for k in range(len(f))],
                   "o", ms=2.5, color=MUTED, alpha=0.8)
            b.plot(f.matched_percentile.mean(), i, "D", ms=5, mfc=INK if r.theory else "white", mec=INK, mew=1)
        else:
            b.text(0.5, i, "too rare to embed", ha="center", va="center", fontsize=6, color=MUTED, style="italic",
                   bbox=dict(fc="white", ec="none", pad=1.5))
    a.axvline(1, color=MUTED, lw=0.8, ls="--"); b.axvline(0.5, color=MUTED, lw=0.8, ls="--")
    a.set_xscale("log"); a.set_xticks([0.5, 1, 2, 4]); a.set_xticklabels(["0.5", "1", "2", "4"]); a.minorticks_off()
    a.set_yticks(range(len(order))); a.set_yticklabels(order, color=INK)
    a.set_xlabel("Salience: Imvo / Isigidimi rate\n(95% issue-bootstrap CI)")
    b.set_xlim(-0.02, 1.02); b.set_xlabel("Framing: shift rank among 50\nfrequency-matched words (0-1)")
    a.set_title("(a) How often", loc="left", fontsize=7.5, color=INK); b.set_title("(b) How differently", loc="left", fontsize=7.5, color=INK)
    b.tick_params(axis="y", left=False)
    a.plot([], [], "o", mfc=INK, mec=INK, label="political concept"); a.plot([], [], "o", mfc="white", mec=INK, label="neutral control")
    a.legend(frameon=False, fontsize=6, loc="lower right")
    fig.savefig(OUT / "fig_two_part.png", bbox_inches="tight")


def global_null():
    raw = np.load(OUT / "framing_raw.npz", allow_pickle=True)
    real, null, probe = raw["real"], raw["null"], raw["probe"]
    nm, rm = null[:, probe].mean(1), real[:, probe].mean(1)
    fig, ax = plt.subplots(figsize=(3.3, 1.9), dpi=200)
    ax.hist(nm, bins=np.linspace(0.47, 0.59, 25), color="#c3c2b7", edgecolor="white", lw=0.5)
    for v in rm:
        ax.axvline(v, color=INK, lw=0.9)
    ax.set_xlim(0.46, 0.65)
    ax.set_ylim(0, 29)
    ax.text(0.578, 27, "real: 10 Imvo draws", fontsize=6.5, color=INK, va="top")
    ax.text(0.465, 27, "permutation null\n(50 permutations)", fontsize=6.5, color=MUTED, va="top")
    ax.set_xlabel("Mean neighbourhood shift (1 - Jaccard@10) over probe words"); ax.set_ylabel("permutations")
    fig.savefig(OUT / "fig_global_null.png", bbox_inches="tight")


def fertility():
    d = pd.read_csv(OUT / "fertility.csv")
    cond = ["modern", "modern, historical spelling", "Isigidimi 1884-88", "Imvo 1884-88 (draw)"]
    labels = ["(a) modern", "(b) modern,\nold spelling", "(c) Isigidimi", "(c) Imvo"]
    cols = {"xlm-roberta-base": BLUE, "serengeti": ORANGE, "mbert-cased": AQUA}
    fig, ax = plt.subplots(figsize=(3.3, 2.1), dpi=200)
    for tok, c in cols.items():
        s = d[d.tokenizer == tok].set_index("condition").loc[cond, "subwords_per_100_chars"]
        pct = 100 * (s / s.iloc[0] - 1)
        ax.plot(range(4), pct.values, "-o", color=c, lw=1.2, ms=4)
        ax.text(3.08, pct.values[-1], {"xlm-roberta-base": "XLM-R", "serengeti": "Serengeti", "mbert-cased": "mBERT"}[tok],
                fontsize=6.5, color=INK, va="center")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=6); ax.set_xlim(-0.2, 4.0)
    ax.set_ylabel("subwords per character,\n% change vs modern")
    fig.savefig(OUT / "fig_fertility.png", bbox_inches="tight")


if __name__ == "__main__":
    two_part(); global_null(); fertility(); print("figures saved")
