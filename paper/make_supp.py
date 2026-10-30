"""Write supp.tex: supplementary tables (stems and matched forms; secondary framing tests across settings;
era drift).  Not part of the two-page limit; appended after the checklist."""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import CONTROL, EXPLORATORY, OUT, THEORY, load_pages  # noqa: E402

esc = lambda s: s.replace("_", "\\_")
# Table S1 needs word counts from the raw data; they are cached in outputs/stem_forms.csv so the
# supplement can be rebuilt without the data (e.g. by a marker who only has the code and outputs).
S1_CACHE = OUT / "stem_forms.csv"
try:
    pages = load_pages()
    counts = Counter(w for ss in pages.sents for s in ss for w in s)
    s1 = []
    for kind, table in (("political", THEORY), ("neutral", CONTROL), ("exploratory", EXPLORATORY)):
        for name, stems in table.items():
            forms = Counter({w: c for w, c in counts.items() if any(s in w for s in stems)})
            s1.append({"concept": name, "kind": kind, "stems": ", ".join(stems), "n_forms": len(forms),
                       "top_forms": ", ".join(f"{w} ({c})" for w, c in forms.most_common(7))})
    pd.DataFrame(s1).to_csv(S1_CACHE, index=False)
except FileNotFoundError:
    s1 = pd.read_csv(S1_CACHE).to_dict("records")

lines = ["\\clearpage", "\\section*{Supplementary material}", "\\footnotesize",
         "\\subsection*{S1. Concepts, stems and the word forms they match (1 Nov 1884--1888, both papers)}",
         "Stems are matched as substrings of surface forms; following \\citet{schoots2023b} we inspected forms for false positives (our first \\emph{law} stem also matched \\emph{inteto}, speech, and was dropped) but did not hand-filter them, so run-together forms (e.g. \\emph{rulumenteningomsebenzi}) and missed variants remain. "
         "All stems, forms and glosses await review by a first-language isiXhosa speaker; e.g.\\ \\emph{rulumente} is glossed `government' here but `parliament' by Schoots et al.\\ (2023a, p.~62). Political concepts were fixed before testing; religion was suggested by the classifier and is exploratory.\\\\[2pt]",
         "\\begin{tabular}{@{}p{1.9cm}p{1.6cm}p{2.0cm}p{9.6cm}@{}}", "\\toprule", "Concept & Type & Stems & Most frequent matched forms (count; number of distinct forms) \\\\ \\midrule"]
for r in s1:
    lines.append(f"{r['concept']} & {r['kind']} & {r['stems']} & {r['top_forms']}; {r['n_forms']} forms \\\\")
lines += ["\\bottomrule", "\\end{tabular}", ""]

SETTINGS = [("window 1 (tuned)", ""), ("window 1, forms joined", "_segnorm"), ("window 2", "window2/%s"), ("window 2, forms joined", "window2/%s_segnorm"),
            ("window 5", "_w5"), ("window 5, forms joined", "_w5_segnorm")]


def load(tag):
    if tag.startswith("window2/"):
        path = OUT / "window2" / f"framing_matched_tests{tag.split('%s')[1] if '%s' in tag else ''}.csv"
    else:
        path = OUT / f"framing_matched_tests{tag}.csv"
    return pd.read_csv(path).set_index("test") if path.exists() else None


ordn = lambda x: f"{100 * x:.0f}"
fp = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
lines += ["\\subsection*{S2. Secondary framing tests across settings}",
          "Percentile of the observed shift among 2000 frequency-matched word sets, and one-sided $p$ (uncorrected). "
          "Jaccard = shift in a word's ten nearest neighbours (neighbour overlap; \\citealp{gonen2020}). P+V = \\emph{palamente} + \\emph{voti}; P+V+R adds \\emph{rulumente} (uncertain meaning). Directional $\\Delta$: Isigidimi minus Imvo, "
          "religious minus political similarity (in $z$-score units), political concepts, two-sided $p$ against matched sets; the last columns give the control \\emph{house}. "
          "Three settings survive Bonferroni over five ($p<0.01$): window 1, window 2, and window 2 with forms joined. Joining partly disjunctive forms (\\emph{uku ba}) leaves the global profile gap unchanged. Matched sets are re-drawn for each test; without \\emph{land} the pooled set is smaller, so lower power may contribute to the higher $p$.\\\\[2pt]",
          "\\begin{tabular}{@{}lcccccccccc@{}}", "\\toprule",
          "Setting & P+V profile & P+V Jaccard & P+V+R profile & P+V+R Jaccard & $\\Delta$ & $\\Delta$ $p$ & $p$ w/o land & house $\\Delta$ & house $p$ \\\\ \\midrule"]
for label, tag in SETTINGS:
    t = load(tag if not tag.startswith("window2") else tag)
    if t is None:
        continue
    cell = lambda test: (f"{ordn(t.loc[test].percentile)} ($p$={t.loc[test].p:.3f})" if test in t.index else "n/a")
    d = t.loc["directional delta: all political concepts"]
    lines.append(f"{label} & {cell('profile shift: palamente+voti (parliament, vote)')} & {cell('Jaccard shift: palamente+voti (parliament, vote)')} & "
                 f"{cell('profile shift: palamente+voti+rulumente (adds government)')} & {cell('Jaccard shift: palamente+voti+rulumente (adds government)')} & "
                 f"{d.obs:+.2f} & {fp(d.p)} & {fp(t.loc['directional delta: political concepts without land'].p)} & {t.loc['directional delta: control:house'].obs:+.2f} & {fp(t.loc['directional delta: control:house'].p)} \\\\")
lines += ["\\bottomrule", "\\end{tabular}", ""]

era = pd.read_csv(OUT / "era_drift.csv")
lines += ["\\subsection*{S3. Era drift: does a concept's rate move across eras within one paper?}",
          "Per-million-token rates and ratios (later/earlier, or Imvo/Isigidimi) with issue-bootstrap 95\\% CIs.\\\\[2pt]",
          "\\begin{tabular}{@{}llrrl@{}}", "\\toprule", "Concept & Comparison & First & Second & Ratio (95\\% CI) \\\\ \\midrule"]
for r in era[~era.comparison.str.contains("main")].itertuples():
    lines.append(f"{r.concept} & {r.comparison} & {r.per_M_first:.0f} & {r.per_M_second:.0f} & {r.ratio:.2f} ({r.ci_lo:.2f}--{r.ci_hi:.2f}) \\\\")
lines += ["\\bottomrule", "\\end{tabular}", ""]

import json
anc = json.load(open(OUT / "anchors.json"))
tex = lambda ws: ", ".join("\\emph{%s}" % w for w in ws)
lines += ["\\subsection*{S4. Anchors for the directional test}",
          "Anchors are the 15 top-PMI neighbours of each seed pair in held-out text (%s), frozen before testing. "
          "Religious (%d words; seeds \\emph{tixo}, \\emph{kristu}): %s. Political (%d words; seeds \\emph{palamente}, \\emph{voti}): %s. "
          "Seed words cannot themselves be tested (Fig.~S1). Political anchors include \\emph{endlu} (`in the house'), "
          "which might have favoured the neutral control \\emph{house}; re-running without \\emph{endlu} leaves both the political lean and \\emph{house}'s lean in place (outputs/framing\\_matched\\_tests\\_noendlu.csv), and \\emph{ngalomteto}, which contains the \\emph{law} stem. Glosses are unverified: two religious anchors (\\emph{ukumkanikazi}, \\emph{inkosazana}) may reflect royal rather than religious register, and several anchors on both sides (\\emph{lika}, \\emph{buka}; \\emph{nawupina}, \\emph{kwabase}, \\emph{izayo}) may be grammatical rather than topical, consistent with narrow-window PMI capturing grammar. All await speaker review." % (
              anc["derived_from"], len(anc["religious"]), tex(anc["religious"]), len(anc["political"]), tex(anc["political"])),
          "\\normalsize", ""]
lchk = json.loads((OUT / "letters_check.json").read_text())
ls = pd.read_csv(OUT / "letters_salience.csv")
lines += ["\\subsection*{S5. Readers' letters versus the rest of each paper}",
          "Letters are detected by src/12\\_letters.py: a letter opens by addressing the editor (vocative \\emph{mhleli} or \\emph{mcokeli}, "
          "not \\emph{u/ku/ngu mhleli}, which refer to him) or by asking for insertion (\\emph{ndifakele}, \\emph{kaundifakele}), and ends at the "
          "signature (a run of very short lines). A hand check of %d random detections found %d genuine letters; the errors were %s. "
          "Recall was not measured. Imvo is restricted to the months in which Isigidimi has text; CIs from 4000 issue resamples; "
          "rates are not tested against matched sets. Hits = concept uses.\\\\[2pt]" % (
              lchk["checked"], lchk["genuine_letters"], "; ".join(lchk["false_positives"]).replace("'", "`", 1)),
          "\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
          "Part & Concept & Isig.\\ tokens & Isig.\\ hits & Imvo tokens & Imvo hits & Imvo/Isig. & 95\\% CI \\\\ \\midrule"]
for _, r in ls.iterrows():
    lines.append(f"{'letters' if r.part == 'letters' else 'rest of paper'} & {r.concept} & {r.tokens_isig:,} & {r.hits_isig} & "
                 f"{r.tokens_imvo:,} & {r.hits_imvo} & {r.ratio:.2f} & {r.ci_lo:.2f}--{r.ci_hi:.2f} \\\\")
lines += ["\\bottomrule", "\\end{tabular}", ""]

sen = pd.read_csv(OUT / "land_senses.csv")
lines += ["\\subsection*{S6. Which sense of \\emph{umhlaba}?}",
          "For every use of the land stem (\\emph{mhlaba}) we look at the 10 tokens either side. Religious cues: \\emph{izulu}, \\emph{lizulu}, "
          "\\emph{nezulu}, and words containing \\emph{zulwini}, \\emph{tixo}, \\emph{kristu}, \\emph{yesu}, \\emph{msindisi}, \\emph{ngcwele} "
          "(bare \\emph{zulu} is excluded: \\emph{kwa zulu} is Zululand). Property/political cues: words containing \\emph{taitile}, \\emph{mteto}, "
          "\\emph{rulumente}, \\emph{palamente}, \\emph{mantyi}, \\emph{fama}, \\emph{voti}, \\emph{nyulo}, \\emph{rafu}, \\emph{lokishi}. "
          "Most uses have neither cue. Example lines for a speaker to check are in outputs/land\\_kwic.md.\\\\[2pt]",
          "\\begin{tabular}{@{}llrrr@{}}", "\\toprule", "Paper & Cue & Uses & Share & 95\\% CI \\\\ \\midrule"]
for _, r in sen.iterrows():
    lines.append(f"{'Isigidimi' if r.paper == 'ISIG' else 'Imvo'} & {r.cue} & {r.uses} & {100 * r.share:.1f}\\% & "
                 f"{100 * r.ci_lo:.1f}--{100 * r.ci_hi:.1f}\\% \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "",
          ]
ec = pd.read_csv(OUT / "editor_control.csv")
lines += ["\\subsection*{S7. Same editor, different control; same control, different editor}",
          "Cells (editors after Schoots, 2023b, p.~71): Jabavu edits Isigidimi (Jan 1881--Oct 1884); Jabavu edits Imvo (Nov 1884--1888); "
          "Gqoba edits Isigidimi (Nov 1884--1888); Makiwane edits Isigidimi (1876--80; Stewart until 1876). Ratios of rates per million tokens; 95\\% CIs from 4000 "
          "issue resamples; $p$: two-sided, against 2000 matched sets of random words (src/15\\_editor\\_control.py). Every contrast is also a contrast in time; "
          "\\emph{rule} has only 9 uses in Jabavu's Isigidimi, so its large ratios are not interpreted.\\\\[2pt]",
          "\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
          "Concept & \\multicolumn{3}{c}{Jabavu: Imvo / Isigidimi} & \\multicolumn{3}{c}{Isigidimi: Gqoba / Jabavu} & \\multicolumn{3}{c}{Isigidimi: 1881--84 / 1876--80} \\\\",
          " & ratio & 95\\% CI & $p$ & ratio & 95\\% CI & $p$ & ratio & 95\\% CI & $p$ \\\\ \\midrule"]
for c in ec.concept.unique():
    cells = []
    for lab in ("control (same editor)", "editor (same control)", "time within Isigidimi (1881-84 vs 1876-80)"):
        r = ec[(ec.concept == c) & (ec.contrast == lab)].iloc[0]
        cells.append(f"{r.ratio:.2f} & {r.ci_lo:.2f}--{r.ci_hi:.2f} & " + ("$<$0.001" if r.matched_p < 0.001 else f"{r.matched_p:.3f}"))
    lines.append(f"{c} & " + " & ".join(cells) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "",
          ]
tg = pd.read_csv(OUT / "tuning_grid.csv")
abl = pd.read_csv(OUT / "ablation_ft_vs_w2v.csv")
best = tg.loc[tg.groupby("window").stability.idxmax()].sort_values("window")
hp = json.loads((OUT / "hparams.json").read_text())
lines += ["\\subsection*{S8. Hyper-parameter search for the embeddings}",
          "fastText skip-gram, tuned on text outside the test window (Isigidimi 1876--83, Imvo 1889--94) so that tuning never sees the "
          "compared years. Criterion: seed stability, the mean top-10 nearest-neighbour overlap between two runs that differ only in "
          "random seed \\citep{antoniak2018}; unstable embeddings would turn noise into apparent shifts. %d settings over dimension "
          "(%s), window (%s) and epochs (%s), searched in rounds that extended the grid past each optimum. Chosen: dimension %d, "
          "window %d, %d epochs. Stability is highest at the narrowest window, which favours grammar over topic; the paper therefore "
          "also reports windows 2 and 5 (Table S2). Best setting per window:\\\\[2pt]" % (
              len(tg), ", ".join(map(str, sorted(tg.vector_size.unique()))), ", ".join(map(str, sorted(tg.window.unique()))),
              ", ".join(map(str, sorted(tg.epochs.unique()))), hp["vector_size"], hp["window"], hp["epochs"]),
          "\\begin{tabular}{@{}rrrrrr@{}}", "\\toprule",
          "Window & Dimension & Epochs & Isigidimi 1876--83 & Imvo 1889--94 & Mean stability \\\\ \\midrule"]
for _, r in best.iterrows():
    lines.append(f"{int(r.window)} & {int(r.vector_size)} & {int(r.epochs)} & {r.stab_isig_1876_83:.3f} & {r.stab_imvo_1889_94:.3f} & {r.stability:.3f} \\\\")
w2v, ft = abl[abl.model == "w2v"].stability.mean(), abl[abl.model == "ft"].stability.mean()
lines += ["\\bottomrule", "\\end{tabular}\\\\[2pt]",
          f"At the chosen setting, word2vec reaches a stability of {w2v:.2f} against fastText's {ft:.2f}, so fastText is used throughout.", "",
          ]
fe = pd.read_csv(OUT / "fertility.csv")
conds = ["modern", "modern, historical spelling", "modern politics/business", "Isigidimi 1884-88", "Imvo 1884-88 (draw)"]
labels = {"modern": "Modern news", "modern, historical spelling": "Modern news, old spelling",
          "modern politics/business": "Modern news, politics/business only", "Isigidimi 1884-88": "Isigidimi 1884--88",
          "Imvo 1884-88 (draw)": "Imvo 1884--88 (matched draw)"}
lines += ["\\subsection*{S9. Tokenizer cost}",
          "Each condition is a random sample of about 100{,}000 words. Modern text is the isiXhosa section of MasakhaNEWS \\citep{adelani2023}. "
          "The old-spelling condition is the same modern sample with the aspiration marks of modern orthography removed by fixed rules "
          "(\\emph{th}$\\to$\\emph{t}, \\emph{kh}$\\to$\\emph{k}, \\emph{ph}$\\to$\\emph{p}, \\emph{rh}$\\to$\\emph{r}, \\emph{ch}$\\to$\\emph{c}, "
          "\\emph{qh}$\\to$\\emph{q}, \\emph{xh}$\\to$\\emph{x}; src/05\\_fertility.py), so only spelling changes and the words stay the same: "
          "it is compared per word. Historical text differs in word division, so it is compared per 100 characters. The politics/business "
          "sample checks genre within modern news: its cost is close to all modern news, while both 1880s papers cost more, so the historical "
          "penalty is not simply a news-topic effect, though 1880s and modern newspapers still differ in genre. 95\\% CIs from bootstrap "
          "resampling are within $\\pm$0.01 subwords per word.\\\\[2pt]",
          "\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
          "Condition & Words & \\multicolumn{2}{c}{XLM-R} & \\multicolumn{2}{c}{Serengeti} & \\multicolumn{2}{c}{mBERT} \\\\",
          " & & per word & per 100 ch. & per word & per 100 ch. & per word & per 100 ch. \\\\ \\midrule"]
for c in conds:
    row = [labels[c], f"{int(fe[(fe.condition == c)].words.iloc[0]):,}"]
    for tok in ("xlm-roberta-base", "serengeti", "mbert-cased"):
        r = fe[(fe.condition == c) & (fe.tokenizer == tok)].iloc[0]
        row += [f"{r.fertility:.3f}", f"{r.subwords_per_100_chars:.2f}"]
    lines.append(" & ".join(row) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "",
          "\\renewcommand{\\thefigure}{S\\arabic{figure}}\\setcounter{figure}{0}",
          "\\begin{figure}[h]\\centering\\includegraphics[width=3.45in]{fig_framing.png}",
          "\\caption{Profile-shift percentile of each concept among matched sets (markers as in Fig.~1, left); "
          "`too rare' = too few occurrences, `seed word' = anchor source, untestable.}\\end{figure}",
          "\\begin{figure}[h]\\centering\\includegraphics[width=2.3in]{fig_parliament_by_year.png}",
          "\\caption{Parliamentary vocabulary per million tokens; it enters Isigidimi in 1883 (light blue: twice-monthly issues, "
          "Schoots et al., 2023a, p.~59).}\\end{figure}", ""]
(Path(__file__).resolve().parent / "supp.tex").write_text("\n".join(lines))
print("wrote supp.tex")
