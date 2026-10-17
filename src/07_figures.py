"""Figures for the write-up: ONE plot per image, each drawn at the size it is used in the paper
(no rescaling): half-page plots 3.45 x 2.5 in, third-page plots 2.3 x 1.8 in."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT

INK, MUTED = "#0b0b0b", "#52514e"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots 1-2 (Isigidimi, Imvo)
plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
MARK = {"theory": "o", "control": "s", "exploratory": "D"}


def _salience_table():
    sal = pd.read_csv(OUT / "salience.csv")
    sal["group"] = sal.concept.str.split(":").str[0]
    sal["name"] = sal.concept.str.split(":").str[1]
    return sal, sal.sort_values("ratio")["name"].tolist()          # bottom -> top


def salience():
    sal, order = _salience_table()
    fig, ax = plt.subplots(figsize=(3.45, 1.85), layout="constrained")
    for i, n in enumerate(order):
        r = sal[sal.name == n].iloc[0]
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=MUTED, lw=1)
        ax.plot(r.ratio, i, MARK[r.group], ms=5, mfc=INK if r.pseudo_p < 0.05 else "white", mec=INK, mew=1)
    ax.axvline(1, color=MUTED, lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.minorticks_off(); ax.set_xlim(0.13, 11)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    ax.set_xlabel("Imvo / Isigidimi rate ratio (95% issue-bootstrap CI)")
    handles = [plt.Line2D([], [], ls="", marker=m, mfc=INK, mec=INK, label=l) for l, m in
               (("political concept", "o"), ("neutral control", "s"), ("religion (exploratory)", "D"))]
    handles.append(plt.Line2D([], [], ls="", marker="o", mfc="white", mec=INK, label="open: within matched null"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=5.5, handletextpad=0.3, labelspacing=0.25)
    fig.savefig(OUT / "fig_salience.png")


def framing():
    sal, order = _salience_table()
    fc = pd.read_csv(OUT / "framing_concepts.csv")
    fc = fc[fc.measure == "profile"].set_index("concept")
    fig, ax = plt.subplots(figsize=(3.45, 1.85), layout="constrained")
    for i, n in enumerate(order):
        g = sal[sal.name == n].iloc[0].group
        if n in fc.index:
            f = fc.loc[n]
            ax.plot(f.percentile, i, MARK[g], ms=5, mfc=INK if f.p_greater < 0.05 else "white", mec=INK, mew=1)
        else:
            ax.text(0.5, i, "too rare / seed word", ha="center", va="center", fontsize=5.5, color=MUTED, style="italic",
                    bbox=dict(fc="white", ec="none", pad=1.2))
    ax.axvline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    ax.set_xlabel("Percentile of profile shift among 2000\nfrequency-matched word sets")
    fig.savefig(OUT / "fig_framing.png")


def yearly(concept, fname, labels):
    d = pd.read_csv(OUT / "salience_yearly.csv")
    fig, ax = plt.subplots(figsize=(2.3, 1.2), layout="constrained")
    for paper, col in (("ISIG", BLUE), ("IMVO", ORANGE)):
        g = d[(d.paper == paper) & (d.concept == concept)].set_index("year").reindex(range(1870, 1895))
        ax.fill_between(g.index, g.ci_lo, g.ci_hi, color=col, alpha=0.15, lw=0)
        ax.plot(g.index, g.per_M, "-o", color=col, lw=1.0, ms=2)
    ax.axvspan(1883.5, 1884.75, color=BLUE, alpha=0.12, lw=0)      # Isigidimi fortnightly, Jul 1883-Sep 1884 (Schoots et al. 2023, p. 59)
    ax.text(*labels["ISIG"], "Isigidimi", fontsize=7.5, color=INK)
    ax.text(*labels["IMVO"], "Imvo", fontsize=7.5, color=INK)
    ax.set_xlim(1869, 1896); ax.set_xticks([1870, 1880, 1890]); ax.tick_params(labelsize=7.5)
    ax.set_ylabel("per million tokens", fontsize=7.5)
    fig.savefig(OUT / fname)


def tokenizer():
    """Two clean contrasts: spelling only, per WORD (modern text vs the same words with aspiration
    stripped: same words, so no unit artefact), and historical text, per CHARACTER (word division
    differs between modern and historical text, so words are not comparable)."""
    f = pd.read_csv(OUT / "fertility.csv")
    get = lambda tok, cond, col: float(f[(f.tokenizer == tok) & (f.condition == cond)][col].iloc[0])
    fig, ax = plt.subplots(figsize=(2.3, 1.2), layout="constrained")
    for tok, lab, m_, dx in (("serengeti", "Serengeti", "s", -0.18), ("xlm-roberta-base", "XLM-R", "o", 0.0), ("mbert-cased", "mBERT", "^", 0.18)):
        spell = 100 * (get(tok, "modern, historical spelling", "fertility") / get(tok, "modern", "fertility") - 1)
        hist = 100 * (get(tok, "Isigidimi 1884-88", "subwords_per_100_chars") / get(tok, "modern", "subwords_per_100_chars") - 1)
        ax.plot([dx, 1 + dx], [spell, hist], ls="", marker=m_, color=INK, ms=4.5, label=lab)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["spelling only\n(per word)", "historical text\n(per character)"], fontsize=6.5)
    ax.set_xlim(-0.5, 1.5); ax.tick_params(labelsize=7.5, axis="y")
    ax.set_ylabel("% vs modern", fontsize=7.5)
    ax.legend(loc="upper left", frameon=False, fontsize=6.5, handletextpad=0.2, labelspacing=0.2, borderaxespad=0.1)
    fig.savefig(OUT / "fig_tokenizer_cost.png")


def global_null():
    raw = np.load(OUT / "framing_raw.npz", allow_pickle=True)
    real, null, probe = raw["real_profile"], raw["null_profile"], raw["probe"]
    nm, rm = null[:, probe].mean(1), real[:, probe].mean(1)
    fig, ax = plt.subplots(figsize=(3.45, 2.0), layout="constrained")
    lo, hi = min(nm.min(), rm.min()), max(nm.max(), rm.max())
    ax.hist(nm, bins=np.linspace(lo - 0.01, hi + 0.01, 40), color="#c3c2b7", edgecolor="white", lw=0.5)
    for v in rm:
        ax.axvline(v, color=INK, lw=0.9)
    top = ax.get_ylim()[1]
    ax.set_xlim(lo - 0.02, hi + 0.075); ax.set_ylim(0, top * 1.35)
    ax.text(lo - 0.012, top * 1.3, "permutation null\n(50 permutations)", fontsize=6.5, color=MUTED, va="top", ha="left")
    ax.text(hi + 0.008, top * 1.3, "real: 10\nImvo draws", fontsize=6.5, color=INK, va="top", ha="left")
    ax.set_xlabel("Mean profile shift (1 - r) over probe words"); ax.set_ylabel("permutations")
    fig.savefig(OUT / "fig_global_null.png")


if __name__ == "__main__":
    salience(); framing()
    yearly("religion", "fig_religion_by_year.png", {"ISIG": (1885.5, 3400), "IMVO": (1888.5, 1300)})
    yearly("parliament", "fig_parliament_by_year.png", {"ISIG": (1870.5, 400), "IMVO": (1890.5, 1250)})
    tokenizer(); global_null()
    print("figures saved")
