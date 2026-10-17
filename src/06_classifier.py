"""Supporting evidence: can a classifier tell Isigidimi from (date-matched) Imvo text, and what does it use?

  * Units: ~150-token chunks; every chunk carries its issue date as a group, so no issue is
    ever split across train and test (grouped nested cross-validation).
  * Outer loop: 5-fold GroupKFold = test; inner loop: 3-fold GroupKFold = tuning of C.
  * Feature sets (to separate framing from orthography and transcription conventions):
      words  : word uni+bigrams (min_df=3)
      chars  : character 2-5-grams (captures spelling, word-division and transcription habits)
      shared : unigrams restricted to words with >=20 uses in BOTH papers, tokens <=2 chars removed
      clean  : shared minus masthead / dateline / title tokens (the first run showed these leak the label)
  * Permutation baseline: issue-level label shuffles (30), same pipeline with C fixed to the modal choice.
  * Repeated over 3 independent Imvo draws.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline

from common import OUT, count_tokens, draw_imvo_subsample, flatten, load_pages

CHUNK = 150
# Tokens that identify the paper or town rather than say anything (found in the top features of the
# first run: newspaper titles, the Lovedale / King William's Town / East London datelines, the
# weekday header, the spelling Xosa/Xhosa, and English courtesy titles).
MASTHEAD = {"isigidimi", "sigidimi", "samaxosa", "xosa", "xhosa", "imvo", "mvo", "zabantsundu", "ndu",
            "ontsundu", "lovedale", "ngolwesi", "qonce", "komani", "bhayi", "town", "sir", "mr"}
GRID = {"logisticregression__C": [0.1, 1, 10]}
N_PERM = 30


def chunks_for(df, label):
    rows = []
    for r in df.itertuples():
        cur, n = [], 0
        for s in r.sents:
            cur.append(" ".join(s)); n += len(s)
            if n >= CHUNK:
                rows.append({"text": " ".join(cur), "label": label, "group": f"{r.paper}_{r.date}"}); cur, n = [], 0
        if cur and n >= CHUNK // 2:
            rows.append({"text": " ".join(cur), "label": label, "group": f"{r.paper}_{r.date}"})
    return rows


def make_pipe(kind, shared=None):
    if kind == "words":
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, token_pattern=r"\S+")
    elif kind == "chars":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, sublinear_tf=True, max_features=200_000)
    else:
        vocab = set(shared) - (MASTHEAD if kind == "clean" else set())
        vec = TfidfVectorizer(vocabulary=sorted(vocab), sublinear_tf=True, token_pattern=r"\S+")
    return make_pipeline(vec, LogisticRegression(max_iter=2000, solver="liblinear"))


def nested_cv(data, kind, shared, labels=None, fixed_c=None, seed=0):
    X, g = data.text.values, data.group.values
    y = data.label.values if labels is None else labels
    scores, aucs, cs = [], [], []
    for tr, te in GroupKFold(5).split(X, y, g):
        pipe = make_pipe(kind, shared)
        if fixed_c is None:
            gs = GridSearchCV(pipe, GRID, cv=GroupKFold(3), scoring="balanced_accuracy", n_jobs=3)
            gs.fit(X[tr], y[tr], groups=g[tr])
            est, c = gs.best_estimator_, gs.best_params_["logisticregression__C"]
        else:
            est, c = pipe.set_params(logisticregression__C=fixed_c).fit(X[tr], y[tr]), fixed_c
        p = est.predict_proba(X[te])[:, 1]
        scores.append(balanced_accuracy_score(y[te], p > 0.5)); aucs.append(roc_auc_score(y[te], p)); cs.append(c)
    return np.array(scores), np.array(aucs), cs


def main():
    pages = load_pages()
    isig_df = pages[pages.paper == "ISIG"]
    rng = np.random.default_rng(1)
    results, tops = [], {}
    for d in range(3):
        imvo_df = draw_imvo_subsample(pages, np.random.default_rng(1000 + d))
        data = pd.DataFrame(chunks_for(isig_df, 1) + chunks_for(imvo_df, 0))   # 1 = Isigidimi
        ci, cm = count_tokens(flatten(isig_df)), count_tokens(flatten(imvo_df))
        shared = {w for w in ci if ci[w] >= 20 and cm.get(w, 0) >= 20 and len(w) > 2}
        print(f"draw {d}: {len(data)} chunks, {data.group.nunique()} issues, shared vocab {len(shared)}")
        for kind in ("words", "chars", "shared", "clean"):
            acc, auc, cs = nested_cv(data, kind, shared)
            row = {"draw": d, "features": kind, "bal_acc": acc.mean(), "bal_acc_sd": acc.std(),
                   "auc": auc.mean(), "modal_C": Counter(cs).most_common(1)[0][0]}
            if d == 0 and kind in ("words", "shared", "clean"):
                groups = data.group.unique()
                null = []
                for _ in range(N_PERM):
                    # shuffle labels at issue level so chunks of one issue keep one (permuted) label
                    lab = dict(zip(groups, rng.permutation([data[data.group == g].label.iloc[0] for g in groups])))
                    null.append(nested_cv(data, kind, shared, labels=data.group.map(lab).values,
                                          fixed_c=row["modal_C"])[0].mean())
                row["perm_null_mean"], row["perm_null_95th"] = float(np.mean(null)), float(np.percentile(null, 95))
                row["perm_p"] = float((1 + sum(n >= acc.mean() for n in null)) / (N_PERM + 1))
            results.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
            if d == 0 and kind in ("words", "shared", "clean"):
                pipe = make_pipe(kind, shared).set_params(logisticregression__C=row["modal_C"]).fit(data.text.values, data.label.values)
                names = np.array(pipe[0].get_feature_names_out()); coef = pipe[1].coef_[0]
                o = np.argsort(coef)
                tops[kind] = {"imvo": names[o[:25]].tolist(), "isigidimi": names[o[::-1][:25]].tolist()}
    pd.DataFrame(results).to_csv(OUT / "classifier.csv", index=False)
    (OUT / "classifier_top_features.json").write_text(json.dumps(tops, indent=1, ensure_ascii=False))
    print(json.dumps(tops, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
