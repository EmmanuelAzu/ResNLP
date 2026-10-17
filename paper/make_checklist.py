"""Build checklist.tex from the course-provided NeurIPS 2024 checklist template.

The questions and guidelines are copied verbatim from NeurIPS_2024_checklist.zip; only the
\\answerTODO / \\justificationTODO macros are replaced by the answers below.
"""
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tex = zipfile.ZipFile(ROOT / "NeurIPS_2024_checklist.zip").read("NeurIPS_2024_checklist/neurips_2024.tex").decode()
body = tex[tex.index("%%% END INSTRUCTIONS %%%"):tex.rindex("\\end{enumerate}") + len("\\end{enumerate}")]
body = body[body.index("\\begin{enumerate}"):]

ANSWERS = [
 ("Yes", "The abstract and introduction state three hypotheses (H1 salience, H2 framing, H3 tokenizer cost) and report what was found, including the negative results (framing is weak and non-specific; the directional lean is not established). Claims about meaning are scoped as unreviewed by a first-language speaker."),
 ("Yes", "The Discussion ends with limitations (two newspapers, one window, 1886 Isigidimi editions missing from the archive, era-confounded held-out windows, no editor metadata in the files used, domain mismatch of the modern news, Serengeti pretraining overlap unverified, speaker review pending) and states the detection floor of the framing test."),
 ("NA", "The paper has no theoretical results."),
 ("Yes", "The Methods give the window, the date-matched subsampling, concept stems, model and hyper-parameter choices, the permutation null, the matched-set tests and the tokenizer conditions. Remaining details (anchor lists, word-joining rules, tokenizer rewrite rules) are in the submitted code and README.txt, which gives the run order; seeds are fixed."),
 ("Yes", "The code is submitted with requirements.txt and README.txt (exact run order). The newspaper texts are the public IsiXhosa Intellectual Traditions archive: the course-provided zip is read by src/01_prepare_data.py and the same texts can be downloaded from the archive DOIs listed in README.txt (Schoots et al., 2023); the modern text is public (MasakhaNEWS) and README.txt gives the download command."),
 ("Yes", "Hyper-parameters were chosen on text outside the test window (Isigidimi 1876-83, Imvo 1889-94) by seed stability over four grid rounds (outputs/tuning_grid.csv), the classifier uses issue-grouped nested cross-validation, and the subsampling and permutation procedures are described."),
 ("Yes", "Salience has issue-level bootstrap CIs; framing uses a permutation null (50 permutations) and 2000 frequency-matched word sets; tokenizer changes have bootstrap CIs (within about 0.5%, stated in the Results). The 15 salience tests and the framing tests are uncorrected; the paper says so, states that neither salience nor framing results survive correction, and rests its argument on effect size, breadth and corroboration. The held-out comparison is era-confounded and is presented as an era-sensitivity check, with within-paper drift as calibration."),
 ("Yes", "All experiments ran on CPU only (no GPU) in roughly three hours in total, including tuning rounds and the earlier framing design that was replaced; the Methods state CPU-only and the runtime, and README.txt gives the four-core setup."),
 ("Yes", "The work uses public-domain historical newspapers and public modern news text, collects no data from people, and flags that interpretation needs first-language-speaker review. To our knowledge it conforms to the Code of Ethics."),
 ("Yes", "The Discussion covers fairness implications (higher tokenizer cost for historical isiXhosa, worst for the African-language model; consequences for planned machine-learning labelling of the archive) and the risk of over-reading statistical shifts as cultural meaning."),
 ("NA", "No models or datasets with misuse risk are released: the outputs are statistics, word lists and code, and the source texts are public-domain archive material."),
 ("Yes", "Assets are credited and licences stated: IsiXIT texts (Schoots et al., 2023; public domain per its authors); MasakhaNEWS (Adelani et al., 2023; AFL-3.0 on its dataset card); XLM-R (MIT) and mBERT (Apache-2.0) per their Hub cards; Serengeti states no licence on its model card and is used only to count tokens; gensim (LGPL) and scikit-learn (BSD) for modelling. Versions are in requirements.txt."),
 ("Yes", "New assets are the analysis code and derived word lists (anchors, word-joining rules, the speaker-check sheet), documented in README.txt; no new text corpus is released."),
 ("NA", "No crowdsourcing or human-subjects research was conducted. A first-language isiXhosa speaker is to be consulted as an expert adviser to check our interpretations; their judgments are not reported as data."),
 ("NA", "No human-subjects research was conducted."),
]
assert len(ANSWERS) == 15
ans_iter, just_iter = iter(ANSWERS), iter(ANSWERS)
body = re.sub(r"\\answerTODO\{\}", lambda m: "\\answer%s{}" % next(ans_iter)[0], body)
body = re.sub(r"\\justificationTODO\{\}", lambda m: next(just_iter)[1].replace("_", "\\_").replace("%", "\\%").replace("&", "\\&"), body)
assert "TODO" not in body
body = re.sub(r"\s*% Replace by \\answerYes\{\}, \\answerNo\{\}, or \\answerNA\{\}\.", "", body)
out = "\\section*{NeurIPS Paper Checklist}\n" + body + "\n"
(Path(__file__).resolve().parent / "checklist.tex").write_text(out)
print("wrote checklist.tex,", len(ANSWERS), "answers")
