"""Tokenizer fertility (subword tokens per word) under three conditions:
  (a) modern isiXhosa news (MasakhaNEWS)
  (b) the SAME modern text rewritten in historical spelling by stripping aspiration marks
  (c) actual historical text (Isigidimi and a date-matched Imvo draw, Nov 1884-1888)
a-b isolates the cost of orthography with content held fixed; b-c is everything else
(vocabulary, register, OCR noise, domain).

Rules (modern -> historical): th->t, kh->k, ph->p, rh->r, ch->c, qh->q, xh->x.
sh, hl, tsh, bh etc. are left alone: not verified against the old orthography and
should be checked with a first-language speaker. Applied to lowercased text in all
conditions so casing is not a confound.

Also reports a sanity check on the rules: the share of historical tokens whose exact
form occurs in the modern vocabulary, before and after stripping.
"""
import re
import warnings

import numpy as np
import pandas as pd

from common import DATA, OUT, clean_to_sentences, draw_imvo_subsample, flatten, load_pages

warnings.filterwarnings("ignore")
N_WORDS = 100_000
B = 1000
# Davlan/afro-xlmr-base ships XLM-R's vocabulary (get_vocab() is identical, checked), so it
# is not listed separately.
TOKENIZERS = [("xlm-roberta-base", "xlm-roberta-base"),
              ("serengeti", "UBC-NLP/serengeti"),
              ("mbert-cased", "bert-base-multilingual-cased")]


class SentencePieceTokenizer:
    """Minimal wrapper so a raw SentencePiece model can be used like a HF tokenizer."""
    def __init__(self, path):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=path)

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": self.sp.encode(list(texts))}


def load_tokenizer(repo):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(repo)
    except Exception:       # Serengeti's tokenizer files fail to convert in current transformers
        from huggingface_hub import hf_hub_download
        return SentencePieceTokenizer(hf_hub_download(repo, "sentencepiece.bpe.model"))


STRIP = [("th", "t"), ("kh", "k"), ("ph", "p"), ("rh", "r"), ("ch", "c"), ("qh", "q"), ("xh", "x")]


def strip_aspiration(word):
    for a, b in STRIP:
        word = word.replace(a, b)
    return word


def modern_sentences(categories=None):
    rows = pd.concat(pd.read_csv(DATA / "modern" / f"xho_{s}.tsv", sep="\t", on_bad_lines="skip")
                     for s in ("train", "dev", "test"))
    if categories:
        rows = rows[rows.category.isin(categories)]
    sents = []
    for t in rows.headline.fillna("").str.cat(rows.text.fillna(""), sep=". "):
        sents.extend(clean_to_sentences(t))
    return sents


def sample_words(sents, n, rng):
    order = rng.permutation(len(sents))
    out, tot = [], 0
    for i in order:
        out.append(sents[i]); tot += len(sents[i])
        if tot >= n:
            break
    return out


def main():
    rng = np.random.default_rng(0)
    pages = load_pages()
    isig = flatten(pages[pages.paper == "ISIG"])
    imvo = flatten(draw_imvo_subsample(pages, np.random.default_rng(1000)))
    modern = modern_sentences()
    modern_pol = modern_sentences(["politics", "business"])
    modern_stripped = [[strip_aspiration(w) for w in s] for s in modern]

    # sanity check of the rewrite rules: do historical forms exist in the modern vocabulary?
    from collections import Counter
    vocab = Counter(w for s in modern for w in s)
    vocab_s = Counter(w for s in modern_stripped for w in s)
    hist = [w for s in isig + imvo for w in s]
    cover = {"historical tokens found in modern vocab (as is)": np.mean([vocab[w] >= 2 for w in hist]),
             "historical tokens found in modern vocab (aspiration stripped)": np.mean([vocab_s[w] >= 2 for w in hist])}
    print(cover)

    conds = {"modern": sample_words(modern, N_WORDS, rng)}
    # (b) must be the same sentences as (a): rebuild from the same sampled indices
    idx_map = {id(s): i for i, s in enumerate(modern)}
    conds["modern, historical spelling"] = [modern_stripped[idx_map[id(s)]] for s in conds["modern"]]
    conds["modern politics/business"] = sample_words(modern_pol, N_WORDS, rng)
    conds["Isigidimi 1884-88"] = sample_words(isig, N_WORDS, rng)
    conds["Imvo 1884-88 (draw)"] = sample_words(imvo, N_WORDS, rng)

    rows = []
    for name, repo in TOKENIZERS:
        try:
            tok = load_tokenizer(repo)
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}: {str(e)[:100]}")
            continue
        for cname, sents in conds.items():
            texts = [" ".join(s) for s in sents]
            ids = tok(texts, add_special_tokens=False)["input_ids"]
            n_sub = np.array([len(i) for i in ids], float)
            n_w = np.array([len(s) for s in sents], float)
            n_c = np.array([sum(map(len, s)) for s in sents], float)
            bs = []
            for _ in range(B):
                r = rng.integers(0, len(sents), len(sents))
                bs.append(n_sub[r].sum() / n_w[r].sum())
            rows.append({"tokenizer": name, "condition": cname, "words": int(n_w.sum()),
                         "fertility": n_sub.sum() / n_w.sum(),
                         "chars_per_word": n_c.sum() / n_w.sum(),
                         "subwords_per_100_chars": 100 * n_sub.sum() / n_c.sum(),
                         "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fertility.csv", index=False)
    print(df.round(3).to_string(index=False))
    for metric in ("fertility", "subwords_per_100_chars"):
        piv = df.pivot(index="tokenizer", columns="condition", values=metric)
        piv["orthography cost (b-a)"] = piv["modern, historical spelling"] - piv["modern"]
        piv["other cost (c-b)"] = piv["Isigidimi 1884-88"] - piv["modern, historical spelling"]
        print("\n" + metric); print(piv.round(3).to_string())
        piv.to_csv(OUT / f"decomposition_{metric}.csv")
    print("example rewrites:", [(" ".join(a[:6]), " ".join(b[:6])) for a, b in
                                zip(conds["modern"][:2], conds["modern, historical spelling"][:2])])


if __name__ == "__main__":
    main()
