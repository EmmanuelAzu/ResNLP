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
pages = load_pages()
counts = Counter(w for ss in pages.sents for s in ss for w in s)

lines = ["\\clearpage", "\\section*{Supplementary material}", "\\footnotesize",
         "\\subsection*{S1. Concepts, stems and the word forms they match (1 Nov 1884--1888, both papers)}",
         "Stems are matched as substrings of surface forms; forms are not hand-filtered, so some may be false positives or miss variants. "
         "All stems, forms and glosses await review by a first-language isiXhosa speaker. Political concepts were fixed before testing; religion was suggested by the classifier and is exploratory.\\\\[2pt]",
         "\\begin{tabular}{@{}p{1.9cm}p{1.6cm}p{2.0cm}p{9.6cm}@{}}", "\\toprule", "Concept & Type & Stems & Most frequent matched forms (count; number of distinct forms) \\\\ \\midrule"]
for kind, table in (("political", THEORY), ("neutral", CONTROL), ("exploratory", EXPLORATORY)):
    for name, stems in table.items():
        forms = Counter({w: c for w, c in counts.items() if any(s in w for s in stems)})
        top = ", ".join(f"{w} ({c})" for w, c in forms.most_common(7))
        lines.append(f"{name} & {kind} & {', '.join(stems)} & {top}; {len(forms)} forms \\\\")
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
lines += ["\\subsection*{S2. Secondary framing tests across settings}",
          "Percentile of the observed shift among 2000 frequency-matched word sets, and one-sided $p$ (uncorrected). "
          "P+V = \\emph{palamente} + \\emph{voti}; P+V+R adds \\emph{rulumente} (uncertain meaning). Directional $\\Delta$: Isigidimi minus Imvo, "
          "religious minus political similarity, political concepts, two-sided $p$ against matched sets.\\\\[2pt]",
          "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
          "Setting & P+V profile & P+V Jaccard & P+V+R profile & P+V+R Jaccard & $\\Delta$ & $\\Delta$ $p$ \\\\ \\midrule"]
for label, tag in SETTINGS:
    t = load(tag if not tag.startswith("window2") else tag)
    if t is None:
        continue
    cell = lambda test: (f"{ordn(t.loc[test].percentile)} ($p$={t.loc[test].p:.3f})" if test in t.index else "n/a")
    d = t.loc["directional delta: all political concepts"]
    lines.append(f"{label} & {cell('profile shift: palamente+voti (parliament, vote)')} & {cell('Jaccard shift: palamente+voti (parliament, vote)')} & "
                 f"{cell('profile shift: palamente+voti+rulumente (adds government)')} & {cell('Jaccard shift: palamente+voti+rulumente (adds government)')} & "
                 f"{d.obs:+.2f} & {d.p:.3f} \\\\")
lines += ["\\bottomrule", "\\end{tabular}", ""]

era = pd.read_csv(OUT / "era_drift.csv")
lines += ["\\subsection*{S3. Era drift: does a concept's rate move across eras within one paper?}",
          "Per-million-token rates and ratios (later/earlier, or Imvo/Isigidimi) with issue-bootstrap 95\\% CIs.\\\\[2pt]",
          "\\begin{tabular}{@{}llrrl@{}}", "\\toprule", "Concept & Comparison & First & Second & Ratio (95\\% CI) \\\\ \\midrule"]
for r in era.itertuples():
    lines.append(f"{r.concept} & {r.comparison} & {r.per_M_first:.0f} & {r.per_M_second:.0f} & {r.ratio:.2f} ({r.ci_lo:.2f}--{r.ci_hi:.2f}) \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\normalsize", ""]
(Path(__file__).resolve().parent / "supp.tex").write_text("\n".join(lines))
print("wrote supp.tex")
