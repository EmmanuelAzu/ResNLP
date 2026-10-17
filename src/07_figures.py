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
    fc = pd.read_csv(OUT / "framing_concepts.csv")
    fc = fc[fc.measure == "profile"].set_index("concept")
    sal["group"] = sal.concept.str.split(":").str[0]
    sal["name"] = sal.concept.str.split(":").str[1]
    order = sal.sort_values("ratio")["name"].tolist()          # bottom -> top
    mk = {"theory": "o", "control": "s", "exploratory": "D"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 3.3), dpi=200, sharey=True, gridspec_kw={"wspace": 0.08})
    for i, n in enumerate(order):
        r = sal[sal.name == n].iloc[0]
        sig = r.pseudo_p < 0.05
        a.plot([r.ci_lo, r.ci_hi], [i, i], color=MUTED, lw=1)
        a.plot(r.ratio, i, mk[r.group], ms=5, mfc=INK if sig else "white", mec=INK, mew=1)
        if n in fc.index:
            f = fc.loc[n]
            b.plot(f.percentile, i, mk[r.group], ms=5, mfc=INK if (f.p_greater < 0.05) else "white", mec=INK, mew=1)
        else:
            b.text(0.5, i, "too rare / seed word", ha="center", va="center", fontsize=6, color=MUTED, style="italic",
                   bbox=dict(fc="white", ec="none", pad=1.5))
    a.axvline(1, color=MUTED, lw=0.8, ls="--"); b.axvline(0.5, color=MUTED, lw=0.8, ls="--")
    a.set_xscale("log"); a.set_xticks([0.25, 0.5, 1, 2, 4]); a.set_xticklabels(["0.25", "0.5", "1", "2", "4"]); a.minorticks_off()
    a.set_yticks(range(len(order))); a.set_yticklabels(order, color=INK)
    a.set_xlabel("Salience: Imvo / Isigidimi rate\n(95% issue-bootstrap CI)")
    b.set_xlim(-0.02, 1.02); b.set_xlabel("Framing: percentile of the concept's profile shift\namong 2000 frequency-matched word sets")
    a.set_title("(a) How often", loc="left", fontsize=7.5, color=INK); b.set_title("(b) How differently", loc="left", fontsize=7.5, color=INK)
    b.tick_params(axis="y", left=False)
    handles = [plt.Line2D([], [], ls="", marker=m_, mfc=INK, mec=INK, label=lab) for lab, m_ in
               (("political concept", "o"), ("neutral control", "s"), ("religion (exploratory)", "D"))]
    handles.append(plt.Line2D([], [], ls="", marker="o", mfc="white", mec=INK, label="open = within matched null (p >= 0.05)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.19))
    fig.savefig(OUT / "fig_two_part.png", bbox_inches="tight")


def global_null():
    raw = np.load(OUT / "framing_raw.npz", allow_pickle=True)
    real, null, probe = raw["real_profile"], raw["null_profile"], raw["probe"]
    nm, rm = null[:, probe].mean(1), real[:, probe].mean(1)
    fig, ax = plt.subplots(figsize=(3.3, 1.9), dpi=200)
    lo, hi = min(nm.min(), rm.min()), max(nm.max(), rm.max())
    ax.hist(nm, bins=np.linspace(lo - 0.01, hi + 0.01, 40), color="#c3c2b7", edgecolor="white", lw=0.5)
    for v in rm:
        ax.axvline(v, color=INK, lw=0.9)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.35)
    ax.text(nm.mean(), top * 1.3, "permutation null\n(50 permutations)", fontsize=6.5, color=MUTED, va="top", ha="center")
    ax.set_xlim(lo - 0.02, hi + 0.075)
    ax.text(hi + 0.008, top * 1.3, "real: 10\nImvo draws", fontsize=6.5, color=INK, va="top", ha="left")
    ax.set_xlabel("Mean profile shift (1 - r) over probe words"); ax.set_ylabel("permutations")
    fig.savefig(OUT / "fig_global_null.png", bbox_inches="tight")


def jabavu():
    d = pd.read_csv(OUT / "salience_yearly.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), dpi=200, sharex=True)
    for ax, concept, title in zip(axes, ("religion", "parliament"), ("(a) religious vocabulary", "(b) parliamentary vocabulary")):
        for paper, col, lab in (("ISIG", BLUE, "Isigidimi"), ("IMVO", ORANGE, "Imvo")):
            g = d[(d.paper == paper) & (d.concept == concept)].set_index("year").reindex(range(1870, 1895))
            ax.fill_between(g.index, g.ci_lo, g.ci_hi, color=col, alpha=0.15, lw=0)
            ax.plot(g.index, g.per_M, "-o", color=col, lw=1.1, ms=2.5)
            last = g.per_M.last_valid_index()
            ax.text(last + 0.4, g.per_M[last], lab, fontsize=6.5, color=INK, va="center")
        ax.axvspan(1880.5, 1883.5, color="#c3c2b7", alpha=0.45, lw=0)
        ax.set_title(title, loc="left", fontsize=7.5, color=INK)
        ax.set_xlim(1869, 1898); ax.set_ylabel("per million tokens")
    axes[0].text(1882, axes[0].get_ylim()[1] * 0.97, "Jabavu's\nyears\n(provisional)", fontsize=5.5, color=MUTED, ha="center", va="top")
    fig.tight_layout()
    fig.savefig(OUT / "fig_jabavu.png", bbox_inches="tight")


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
    two_part(); global_null(); jabavu(); fertility(); print("figures saved")
