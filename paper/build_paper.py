"""Write paper.tex from a template, filling every result number from outputs/*.csv|json so the
text cannot drift from the results.  Run after the experiments:  python paper/build_paper.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def pv(x):
    return "<0.001" if x < 0.001 else f"{x:.3f}"


base = json.loads((OUT / "framing_summary.json").read_text())
seg = json.loads((OUT / "framing_summary_segnorm.json").read_text())
mt = pd.read_csv(OUT / "framing_matched_tests.csv").set_index("test")
ms = pd.read_csv(OUT / "framing_matched_tests_segnorm.csv").set_index("test")
fc = pd.read_csv(OUT / "framing_concepts.csv")
fw = pd.read_csv(OUT / "framing_words.csv")

obs, null = base["global_profile"]["obs"], base["global_profile"]["null_mean"]
gap1 = obs - null
gap2 = seg["global_profile"]["obs"] - seg["global_profile"]["null_mean"]
P2 = "{} shift: palamente+voti (parliament, vote)"
P3 = "{} shift: palamente+voti+rulumente (adds government)"
pr, ja = "profile", "Jaccard"
def ordinal(n):
    n = int(round(n))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


pct = lambda r: ordinal(100 * r.percentile)


def row(tab, pool, m):
    return tab.loc[pool.format(m)]


# window-2 sensitivity (results at the earlier tuned setting, kept in outputs/window2)
w2 = [pd.read_csv(OUT / "window2" / f).set_index("test") for f in ("framing_matched_tests.csv", "framing_matched_tests_segnorm.csv")]
w2rows = [row(t, P2, m) for t in w2 for m in (pr, ja)]
w2lo, w2hi = min(r.percentile for r in w2rows), max(r.percentile for r in w2rows)
w2plo, w2phi = min(r.p for r in w2rows), max(r.p for r in w2rows)

dp_u, dc_u = mt.loc["directional delta: all political concepts"], mt.loc["directional delta: all neutral controls"]
dp_j, dc_j = ms.loc["directional delta: all political concepts"], ms.loc["directional delta: all neutral controls"]
c1 = fc[(fc.measure == "profile") & (fc.forms == 1)]
floor1 = 1.645 * c1.matched_set_sd.mean()
big = fc[fc.measure == "profile"].sort_values("forms", ascending=False).iloc[0]
floor_big = 1.645 * big.matched_set_sd
baseline = float(fw.profile_shift.median())
fcs = pd.read_csv(OUT / "framing_concepts_segnorm.csv")
par_u = fc[(fc.measure == "profile") & (fc.concept == "parliament")].iloc[0]
par_j = fcs[(fcs.measure == "profile") & (fcs.concept == "parliament")].iloc[0]
cat_u = fc[(fc.measure == "profile") & (fc.concept == "cattle")].iloc[0]
cat_j = fcs[(fcs.measure == "profile") & (fcs.concept == "cattle")].iloc[0]

sal = pd.read_csv(OUT / "salience.csv").set_index("concept")
vocab = pd.read_csv(OUT / "vocab_aspiration.csv").set_index("tokenizer")
held = pd.read_csv(OUT / "salience_heldout.csv").set_index("concept")
yr = pd.read_csv(OUT / "salience_yearly.csv")
rel_isig = yr[(yr.paper == "ISIG") & (yr.concept == "religion")].set_index("year").per_M
fert = pd.read_csv(OUT / "fertility.csv")


def spc(tok, cond):
    return float(fert[(fert.tokenizer == tok) & (fert.condition == cond)].subwords_per_100_chars.iloc[0])


def pc(tok, cond):
    return 100 * (spc(tok, cond) / spc(tok, "modern") - 1)


abl = pd.read_csv(OUT / "ablation_ft_vs_w2v.csv")
clf = pd.read_csv(OUT / "classifier.csv")
share = lambda tok: 100 * pc(tok, "modern, historical spelling") / pc(tok, "Isigidimi 1884-88")
sg = lambda x: "0.00" if abs(x) < 0.005 else f"{x:+.2f}".replace("-", "-")
signed = lambda x: f"{x:+.1f}".replace("-", "$-$")

era = pd.read_csv(OUT / "era_drift.csv")
drift = lambda concept, lab: float(era[(era.concept == concept) & (era.comparison == lab)].ratio.iloc[0])
pardrift_hi = float(era[(era.concept == "parliament") & (era.comparison == "Isigidimi: later / earlier era")].ci_hi.iloc[0])
d_pr = [drift(c, l) for c in ("parliament", "religion") for l in ("Isigidimi: later / earlier era", "Imvo: later / earlier era")]
d_gov = [drift("government", l) for l in ("Isigidimi: later / earlier era", "Imvo: later / earlier era")]
pool_cells = [row(t, P2, m) for t in (mt, ms) for m in (pr, ja)]


def wordchg(tok):
    g = lambda c, col: float(fert[(fert.tokenizer == tok) & (fert.condition == c)][col].iloc[0])
    return 100 * (g("modern, historical spelling", "fertility") / g("modern", "fertility") - 1)


def ci_pct(tok, cond_b, col, lo, hi):
    g = lambda c, k: float(fert[(fert.tokenizer == tok) & (fert.condition == c)][k].iloc[0])
    a, bb = g("modern", col), g(cond_b, col)
    ha, hb = (g("modern", hi) - g("modern", lo)) / 2, (g(cond_b, hi) - g(cond_b, lo)) / 2
    return 100 * np.sqrt((ha / a) ** 2 + (hb / bb) ** 2) * bb / a


tci = max([ci_pct(t, "modern, historical spelling", "fertility", "ci_lo", "ci_hi") for t in ("xlm-roberta-base", "serengeti", "mbert-cased")]
          + [ci_pct(t, "Isigidimi 1884-88", "subwords_per_100_chars", "spc_ci_lo", "spc_ci_hi") for t in ("xlm-roberta-base", "serengeti", "mbert-cased")])


def w5_vals():
    """Window-5 sensitivity (unjoined run)."""
    tu = pd.read_csv(OUT / "framing_matched_tests_w5.csv").set_index("test")
    fcw = pd.read_csv(OUT / "framing_concepts_w5.csv")
    par = fcw[(fcw.measure == "profile") & (fcw.concept == "parliament")].iloc[0]
    return {"W5PAR": ordinal(100 * par.percentile), "W5PARP": pv(par.p_greater),
            "W5POOLP": ordinal(100 * row(tu, P2, pr).percentile), "W5POOLJ": ordinal(100 * row(tu, P2, ja).percentile)}


W5 = w5_vals()
tw5 = pd.read_csv(OUT / "framing_matched_tests_w5.csv").set_index("test")
five = [mt, ms, w2[0], w2[1], tw5]
jac5 = [row(t, P2, ja) for t in five]
prof4 = [row(t, P2, pr) for t in five[:4]]
dirs5 = [t.loc["directional delta: all political concepts"] for t in five]
ctl5 = [t.loc["directional delta: all neutral controls"] for t in five]
nol5 = [t.loc["directional delta: political concepts without land"] for t in five]
house5 = [t.loc["directional delta: control:house"] for t in five]
maxc = fc[fc.measure == "profile"].sort_values("forms", ascending=False).iloc[0]
V2 = {
 "JPVLO": ordinal(100 * min(c.percentile for c in jac5)), "JPVHI": ordinal(100 * max(c.percentile for c in jac5)),
 "JPVPLO": pv(min(c.p for c in jac5)), "JPVPHI": pv(max(c.p for c in jac5)),
 "PROFLO": ordinal(100 * min(c.percentile for c in prof4)), "PROFHI": ordinal(100 * max(c.percentile for c in prof4)),
 "DIRLO": f"{min(d.obs for d in dirs5):+.2f}", "DIRHI": f"{max(d.obs for d in dirs5):+.2f}",
 "DIRPHI": pv(max(d.p for d in dirs5)), "CTRLP": f"{min(d.p for d in ctl5):.2f}", "CTRLMAX": f"{max(abs(d.obs) for d in ctl5):.2f}",
 "NOLANDLO": pv(min(d.p for d in nol5)), "NOLANDHI": pv(max(d.p for d in nol5)),
 "HOUSELO": f"{min(d.obs for d in house5):.1f}", "HOUSEHI": f"{max(d.obs for d in house5):.1f}",
 "FLOORN": f"{1.645 * maxc.matched_set_sd:.2f}", "NMAX": str(int(maxc.forms)),
}
V = {
 "OBS": f"{obs:.3f}", "NULL": f"{null:.3f}", "GAP1": f"{gap1:.3f}", "GAP2": f"{gap2:.3f}",
 "P1": pct(row(mt, P2, pr)), "J1": pct(row(mt, P2, ja)), "PP1": pv(row(mt, P2, pr).p), "PJ1": pv(row(mt, P2, ja).p),
 "P2": pct(row(ms, P2, pr)), "J2": pct(row(ms, P2, ja)), "PP2": pv(row(ms, P2, pr).p), "PJ2": pv(row(ms, P2, ja).p),
 "PP3": pv(row(mt, P3, pr).p), "PJ3": pv(row(mt, P3, ja).p), "PP4": pv(row(ms, P3, pr).p), "PJ4": pv(row(ms, P3, ja).p),
 "W2LO": ordinal(100 * w2lo), "W2HI": ordinal(100 * w2hi), "W2PLO": pv(w2plo), "W2PHI": pv(w2phi),
 "PARU": ordinal(100 * par_u.percentile), "PARUP": pv(par_u.p_greater), "PARJ": ordinal(100 * par_j.percentile), "PARJP": pv(par_j.p_greater),
 "CATLO": ordinal(100 * cat_u.percentile), "CATHI": ordinal(100 * cat_j.percentile),
 "DU": sg(dp_u.obs), "DMU": sg(dp_u.matched_mean), "DPU": pv(dp_u.p), "DCU": sg(dc_u.obs),
 "DJ": sg(dp_j.obs), "DMJ": sg(dp_j.matched_mean), "DPJ": pv(dp_j.p), "DCJ": sg(dc_j.obs),
 "FLOOR1": f"{floor1:.2f}", "FLOOR8": f"{floor_big:.2f}", "BASE": f"{baseline:.2f}", "NBIG": ("an " if int(big.forms) in (8, 11, 18) else "a ") + str(int(big.forms)),
 "DW2A": pv(w2[0].loc["directional delta: all political concepts"].p), "DW2B": pv(w2[1].loc["directional delta: all political concepts"].p),
 "SPAR": f"{sal.loc['theory:parliament'].ratio:.1f}", "SPARLO": f"{sal.loc['theory:parliament'].ci_lo:.1f}", "SPARHI": f"{sal.loc['theory:parliament'].ci_hi:.1f}",
 "SPARP": pv(sal.loc['theory:parliament'].pseudo_p), "SREL": f"{sal.loc['exploratory:religion'].ratio:.2f}", "SRELP": pv(sal.loc['exploratory:religion'].pseudo_p),
 "HPAR": f"{held.loc['theory:parliament'].ratio:.1f}", "HPARP": pv(held.loc['theory:parliament'].pseudo_p),
 "HREL": f"{held.loc['exploratory:religion'].ratio:.2f}", "HRELP": pv(held.loc['exploratory:religion'].pseudo_p),
 "GOVM": f"{sal.loc['theory:government'].ratio:.2f}", "GOVH": f"{held.loc['theory:government'].ratio:.2f}",
 "RELPEAK": f"{round(float(rel_isig.max()), -2):,.0f}".replace(",", "{,}"), "RELLOW": f"{round(float(rel_isig.loc[[1882, 1883]].min()), -2):,.0f}",
 "XSP": f"{pc('xlm-roberta-base', 'modern, historical spelling'):.1f}", "XHI": f"{pc('xlm-roberta-base', 'Isigidimi 1884-88'):.1f}",
 "SSP": f"{pc('serengeti', 'modern, historical spelling'):.1f}", "SHI": f"{pc('serengeti', 'Isigidimi 1884-88'):.1f}",
 "MSP": signed(pc("mbert-cased", "modern, historical spelling")), "MHI": f"{pc('mbert-cased', 'Isigidimi 1884-88'):.1f}",
 "XSHARE": f"{share('xlm-roberta-base'):.0f}", "SSHARE": f"{share('serengeti'):.0f}",
 "FTSTAB": f"{abl[abl.model == 'ft'].stability.mean():.2f}", "W2VSTAB": f"{abl[abl.model == 'w2v'].stability.mean():.2f}",
 "VOTEQ": f"{sal.loc[sal.comparison.eq('main') & sal.index.eq('theory:vote'), 'ratio'].iloc[0]:.2f}" if False else f"{sal[sal.comparison == 'main'].loc['theory:vote', 'ratio']:.2f}",
 "SERTH": f"{vocab.loc['serengeti', 'percent']:.1f}", "XLMTH": f"{vocab.loc['xlm-roberta-base', 'percent']:.1f}",
 "PARDRIFTHI": f"{pardrift_hi:.1f}", "PARDRIFT": f"{drift('parliament', 'Isigidimi: later / earlier era'):.1f}",
 "ERAPR": f"{min(d_pr):.1f}--{max(d_pr):.1f}", "VOTEDRIFT": f"{drift('vote', 'Isigidimi: later / earlier era'):.1f}", "GOVDRIFT": f"{min(d_gov):.1f}--{max(d_gov):.2f}",
 "PVLO": ordinal(100 * min(c.percentile for c in pool_cells)), "PVHI": ordinal(100 * max(c.percentile for c in pool_cells)),
 "PVPLO": pv(min(c.p for c in pool_cells)), "PVPHI": pv(max(c.p for c in pool_cells)),
 "XW": f"{wordchg('xlm-roberta-base'):.1f}".replace("-", "$-$"), "SW": f"{wordchg('serengeti'):.1f}", "MW": f"{wordchg('mbert-cased'):.1f}".replace("-", "$-$"),
 "TCI": f"{tci:.1f}", **W5, **V2,
 "CLFACC": f"{clf[clf.features == 'clean'].bal_acc.mean():.2f}",
}
tpl = (ROOT / "paper" / "paper_template.tex").read_text()
for k, v in V.items():
    tpl = tpl.replace("@@" + k + "@@", v)
assert "@@" not in tpl, [x for x in tpl.split("@@")[1:2]]
(ROOT / "paper" / "paper.tex").write_text(tpl)
print({k: V[k] for k in V})
