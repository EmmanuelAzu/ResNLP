What missionary control left in the language of two 1880s isiXhosa newspapers
(Isigidimi sama-Xosa vs Imvo Zabantsundu)  -  COMS4054A / COMS7062A / COMS7066A, Project 2
Authors: Emmanuel Azubuike (2698752), George Ayensu (2722188), Banele Mkhize (2663814)

Question: did missionary control change WHAT the papers printed (salience), HOW they used words
(framing), and does this history still affect how modern tokenizers read the text?

CONTENTS
  src/        the analysis code, numbered in run order (common.py and emb.py are shared helpers)
  outputs/    the results of our run (CSV / JSON / figures), so results can be inspected without
              rerunning; running the scripts regenerates these files
  requirements.txt

SETUP
  Python 3.10+ (developed on 3.13), 4 CPU cores recommended, no GPU needed.
      pip install -r requirements.txt
  DATA: place the course-provided TxtLangFiles1870-1894_TB.zip (IsiXhosa Intellectual Traditions
  text files) in this folder, next to this README.  It is not included because of its size.
  Archive DOIs: Isigidimi https://doi.org/10.25375/uct.22332271 ; Imvo https://doi.org/10.25375/uct.22332268
  Run every command from this folder.

QUICK RUN  (about 10 minutes; reproduces the salience, letters, editor and word-sense results exactly)
  python src/01_prepare_data.py          clean the transcriptions -> data/pages.csv.gz
  python src/04_salience.py              topic rates, bootstrap CIs, matched random-word null  -> outputs/salience*.csv
  python src/04b_yearly_series.py        yearly rates per paper                                 -> outputs/salience_yearly.csv
  python src/04c_era_drift.py            each paper's own drift between eras                    -> outputs/era_drift.csv
  python src/12_letters.py               split pages into readers' letters and the rest         -> outputs/letters_*.csv/.md
  python src/13_letters_salience.py      topic rates in letters vs the rest                     -> outputs/letters_salience.csv
  python src/14_land_senses.py           which sense of 'umhlaba' each paper uses               -> outputs/land_senses.csv, land_kwic.md
  python src/15_editor_control.py        same editor (Jabavu) under both papers; same paper, other editor
                                                                                                -> outputs/editor_control.csv
  python src/07_figures.py               all figures                                            -> outputs/fig_*.png

FULL RUN  (adds the embedding, classifier and tokenizer experiments; a few CPU hours in total)
  python src/02_tune_hparams.py          embedding hyper-parameter search on years outside the comparison
  python src/02_tune_hparams.py --extend   further rounds past each optimum (chosen: dim 400, window 1, 20 epochs)
                                                                                                -> outputs/tuning_grid.csv, hparams.json
  python src/08_anchors.py               freeze the religious / political anchor words (held-out years) -> outputs/anchors.json
  python src/03_framing.py               embedding shift vs shuffled-issue null, directional test   (~17 min)
  python src/10_matched_tests.py         framing tests against frequency-matched word sets
  python src/03_framing.py --segnorm ; python src/10_matched_tests.py _segnorm        joined word forms
  python src/03_framing.py --window 5 --tag _w5 ; python src/10_matched_tests.py _w5  wider window
  python src/03_framing.py --perms-per-draw 1 --drop-anchor endlu --tag _noendlu ; python src/10_matched_tests.py _noendlu
                                                                                      without the anchor 'endlu'
  python src/11_reassign_concepts.py     recompute topic membership after a stem change (no retraining)
  python src/06_classifier.py            issue-grouped nested-CV classifier (suggested the religion topic)
  python src/05_fertility.py             tokenizer cost: modern / old-spelling / historical text (needs MODERN DATA)
  python src/05b_vocab_aspiration.py     share of th/kh/ph pieces in each tokenizer's vocabulary
  python src/09_speaker_sheet.py         review sheet for a first-language isiXhosa speaker -> outputs/speaker_sheet.md
  outputs/window2/ holds the framing run at window 2 (dim 300), kept as a robustness check.

MODERN DATA  (only for 05_fertility.py; needs internet once)
  mkdir -p data/modern && cd data/modern
  for s in train dev test; do curl -L -o xho_$s.tsv \
    https://huggingface.co/datasets/masakhane/masakhanews/resolve/main/data/xho/$s.tsv; done
  The tokenizers (XLM-R, mBERT, Serengeti) are downloaded from the Hugging Face Hub on first use.
  Licences: MasakhaNEWS AFL-3.0; XLM-R MIT; mBERT Apache-2.0; Serengeti states no licence (used only to count tokens).

KEY RESULT FILES
  salience.csv            Imvo/Isigidimi rate ratios, CIs and matched-set p-values per topic
  letters_salience.csv    the same in readers' letters vs the rest of each paper
  editor_control.csv      same-editor and same-paper contrasts (with matched-set p-values)
  framing_matched_tests*.csv   framing tests in each setting; tuning_grid.csv: embedding search
  fertility.csv           tokenizer cost per condition; land_senses.csv: senses of 'umhlaba'
  letters_sample.md / letters_check.json   hand check of the letter detector (28 of 30 genuine)

NOTES
  Random seeds are fixed, and random word sets are drawn from a sorted word list, so reruns give
  identical numbers.  Topic stems (src/common.py) include isiXhosa spelling variants found in the
  data (e.g. Imvo's 'rulumeni' for 'rulumente', government); English words are excluded.
