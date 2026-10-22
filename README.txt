Imvo Zabantsundu vs Isigidimi sama-Xosa: does missionary control show in how political
concepts are used and framed?  (COMS4054A / COMS7062A / COMS7066A, Project 2)

REQUIREMENTS
  Python 3.10+ (developed on 3.13).  pip install -r requirements.txt
  The isiXhosa newspaper data ships in the repository root as TxtLangFiles1870-1894_TB.zip.
  Step 05 needs internet access once: it downloads modern isiXhosa news (MasakhaNEWS, xho)
  and the tokenizers from the Hugging Face Hub. Everything else runs offline.

RUN ORDER  (from the repository root; scripts are in src/, results go to outputs/)
  python src/01_prepare_data.py        build data/pages.csv.gz (cleaned, one row per page)
  python src/02_tune_hparams.py        round 1 of the hyper-parameter search, on text OUTSIDE the test window
  python src/02_tune_hparams.py --extend   rounds 2-4: grid extended past the previous optimum (GRID_EXT is edited
                                       between rounds; rows are merged into outputs/tuning_grid.csv).  Final setting: dim 400,
                                       window 1 (the smallest possible), 20 epochs.  ~10 min each, 4 cores
  python src/08_anchors.py             freeze the religious / political anchor lists (from held-out windows) BEFORE step 03
  python src/03_framing.py             profile-shift / Jaccard / directional tests vs permutation null  (~17 min, 4 cores;
                                       --quick = 2-minute smoke test)
  python src/10_matched_tests.py       calibrate framing against frequency-matched word sets; data-driven word list; leave-one-concept-out directional tests
  python src/11_reassign_concepts.py   recompute concept membership + concept-level matched tests from an existing per-word table (no retraining);
                                       used after the 'law' stem was corrected (the old stem 'nteto' also matched 'intetho', speech)
  python src/04_salience.py            per-million rates, issue-bootstrap CIs, pseudo-concept null, held-out replication,
                                       Jabavu-period contrast
  python src/04b_yearly_series.py      yearly rates (religion / parliament figures)
  python src/04c_era_drift.py          within-paper era drift: calibrates the held-out (era-confounded) comparison
  python src/05_fertility.py           tokenizer fertility incl. the aspiration-stripping control (needs MODERN DATA below)
  python src/12_letters.py            split pages into readers' letters and the rest (hand-check sample: outputs/letters_sample.md)
  python src/13_letters_salience.py   concept rates in letters vs the rest of each paper (outputs/letters_salience.csv)
  python src/14_land_senses.py        which sense of umhlaba each paper uses (outputs/land_senses.csv, land_kwic.md)
  python src/03_framing.py --perms-per-draw 1 --drop-anchor endlu --tag _noendlu ; python src/10_matched_tests.py _noendlu
                                      directional test without the anchor endlu ('in the house')
  python src/05b_vocab_aspiration.py   share of vocabulary entries containing th/kh/ph, per tokenizer
  python src/06_classifier.py          grouped nested-CV classifier (supporting evidence)
  python src/07_figures.py             figures used in the write-up
  python src/09_speaker_sheet.py       sheet for a first-language speaker (outputs/speaker_sheet.md; key in speaker_key.json)
  python src/03_framing.py --segnorm   word-joining control (outputs/*_segnorm.*); run 10_matched_tests.py with the argument _segnorm
  python src/03_framing.py --window 5 --tag _w5 [--segnorm]   window-5 sensitivity (outputs/*_w5*); then 10_matched_tests.py _w5 / _w5_segnorm
  python paper/make_checklist.py       fill the NeurIPS checklist from the course template (paper/checklist.tex)
  python paper/make_supp.py            supplementary tables S1-S3 (paper/supp.tex)
  python paper/build_paper.py          write paper/paper.tex from paper_template.tex, taking every number from outputs/
  (cd paper && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper)

SENSITIVITY
  outputs/window2/ holds the framing results at the earlier tuned setting (dim 300, window 2), kept as a robustness check;
  the paper quotes the pooled percentile range and the directional p-values from it.  outputs/*_w5* hold window 5.
  Tokenizer contrasts: spelling alone is judged per WORD (the stripped text has the same words); historical text per CHARACTER.

MODERN DATA" below)
  python src/06_classifier.py     grouped nested-CV classifier (supporting evidence)
  python src/07_figures.py        figures used in the write-up

ARCHIVE DOIs (Schoots et al., 2023): Isigidimi https://doi.org/10.25375/uct.22332271 ; Imvo https://doi.org/10.25375/uct.22332268

MODERN DATA (for 05_fertility.py)
  mkdir -p data/modern && cd data/modern
  for s in train dev test; do curl -L -o xho_$s.tsv \
    https://huggingface.co/datasets/masakhane/masakhanews/resolve/main/data/xho/$s.tsv; done
  MasakhaNEWS: AFL-3.0 per its Hugging Face dataset card; cite Adelani et al. (2023).
  Other assets: XLM-R (MIT) and mBERT (Apache-2.0) per their Hub cards; Serengeti states no licence on its model
  card and is only used to count tokens; gensim (LGPL), scikit-learn (BSD).

DESIGN IN ONE PARAGRAPH
  Window: 1 Nov 1884 - 1888 (Imvo's first issue is 3 Nov 1884; Isigidimi has no 1886 issues in
  the archive).  Imvo is subsampled month by month to Isigidimi's token count (287k), drawn 10
  times.  The null is a permutation test: issues from both papers are pooled and reassigned to
  two token-balanced groups within strata, a model is trained on each, and the same shift is
  measured.  Because the real groups differ in word frequency and in paper-level style but the
  permuted groups do not, per-word and per-concept framing is judged against frequency-matched
  word sets (2000 random sets), and salience against 2000 frequency-matched pseudo-concepts.
  Salience (how often) and framing (how differently) are reported separately.

PROVISIONAL INPUTS TO CHECK
  * Jabavu's Isigidimi years (1881-83 used in 04_salience / 04b_yearly_series) come from secondary
    web sources, not from Schoots et al. (not reachable from the build environment): confirm them.
  * Spelling rules, concept stems, anchors and neighbour lists have not been checked by a speaker.

FILES
  src/common.py   cleaning, loading, strict window, date-matched subsampling, concept stems
  src/emb.py      fastText/word2vec training, neighbour retrieval, shift and variant measures
  outputs/        csv/json/png results (outputs/cache is git-ignored)
  data/           generated; git-ignored

SEEDS
  Subsampling uses seeds 1000+draw; model seeds are fixed in the scripts.  gensim runs with a
  single worker per model, so results are reproducible for a fixed package version.
