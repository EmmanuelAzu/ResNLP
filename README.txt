Imvo Zabantsundu vs Isigidimi sama-Xosa: does missionary control show in how political
concepts are used and framed?  (COMS4054A / COMS7062A / COMS7066A, Project 2)

REQUIREMENTS
  Python 3.10+ (developed on 3.13).  pip install -r requirements.txt
  The isiXhosa newspaper data ships in the repository root as TxtLangFiles1870-1894_TB.zip.
  Step 05 needs internet access once: it downloads modern isiXhosa news (MasakhaNEWS, xho)
  and the tokenizers from the Hugging Face Hub. Everything else runs offline.

RUN ORDER  (from the repository root; scripts are in src/, results go to outputs/)
  python src/01_prepare_data.py   build data/pages.csv.gz (cleaned, one row per page)
  python src/02_tune_hparams.py   hyper-parameter search on text OUTSIDE the test window  (~10 min, 4 cores)
  python src/03_framing.py        embedding shift vs permutation null                       (~20 min, 4 cores)
                                  add --quick for a 2-minute smoke test
  python src/04_salience.py       per-million rates of concept stems with issue-bootstrap CIs
  python src/05_fertility.py      tokenizer fertility, incl. the aspiration-stripping control
                                  (first download MasakhaNEWS: see "MODERN DATA" below)
  python src/06_classifier.py     grouped nested-CV classifier (supporting evidence)
  python src/07_figures.py        figures used in the write-up

MODERN DATA (for 05_fertility.py)
  mkdir -p data/modern && cd data/modern
  for s in train dev test; do curl -L -o xho_$s.tsv \
    https://huggingface.co/datasets/masakhane/masakhanews/resolve/main/data/xho/$s.tsv; done
  MasakhaNEWS is released for non-commercial research use; cite Adelani et al. (2023).

DESIGN IN ONE PARAGRAPH
  Window: 1 Nov 1884 - 1888 (Imvo's first issue is 3 Nov 1884; Isigidimi has no 1886 issues in
  the archive).  Imvo is subsampled month by month to Isigidimi's token count (287k), drawn 10
  times.  The null is a permutation test: issues from both papers are pooled and reassigned to
  two token-balanced groups within strata, a model is trained on each, and the same shift is
  measured.  Because the real groups differ in word frequency but the permuted groups do not,
  each word is also ranked against frequency-matched control words.  Salience (how often) and
  framing (how differently) are reported separately.

FILES
  src/common.py   cleaning, loading, strict window, date-matched subsampling, concept stems
  src/emb.py      fastText/word2vec training, neighbour retrieval, shift and variant measures
  outputs/        csv/json/png results (outputs/cache is git-ignored)
  data/           generated; git-ignored

SEEDS
  Subsampling uses seeds 1000+draw; model seeds are fixed in the scripts.  gensim runs with a
  single worker per model, so results are reproducible for a fixed package version.
