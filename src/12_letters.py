"""Split each page into readers' letters and everything else (editorials, news, notices).

A letter opens by addressing the editor, e.g. "nkosi yam mhleli ... kaundifakele lamazwana"
("my lord editor ... please insert these few words"), and closes with the writer's signature:
a run of very short lines (name, place).  Notices and editorials instead *refer to* the editor
("ku mhleli", "ngu mhleli", "umhleli"), so the rule keys on the vocative form.
  start = a line of >= 4 tokens that addresses the editor (vocative mhleli / mcokeli, not preceded
          by ku / ngu / ka / kwa) within its first 6 tokens, or asks for insertion (ndifakele, kaundifakele, ...)
          within its first 15 tokens;
  end   = the first run of >= 2 consecutive lines of <= 2 tokens (the signature), the next start,
          a price-list or section line, or 150 lines.
Writes outputs/letters_segments.csv (letter / other tokens per page) and outputs/letters_sample.md,
a random sample for checking by eye (precision is reported in the paper's supplement).
"""
import re

import pandas as pd

from common import OUT, load_pages

MAXLEN = 150
REFER = {"u", "ku", "ngu", "engu", "ka", "kwa", "lika", "we", "wo"}
REQUEST = re.compile(r"(ndifak|kaufak|usifak|wondifak)")
STOP = re.compile(r"^(editorial|isaziso|izaziso|ixabiso|amaxabiso|market)")


def is_start(line):
    if len(line) < 4:
        return False
    for k, t in enumerate(line[:6]):
        if t in ("mhleli", "mcokeli") and (k == 0 or line[k - 1] not in REFER):
            return True
    return any(REQUEST.search(t) or (t.startswith("fakel") and k and line[k - 1] in ("kau", "ka", "undi", "ndi"))
               for k, t in enumerate(line[:15]))


def is_stop(line):
    return bool(STOP.match(" ".join(line))) or sum(t == "to" for t in line) >= 2


def split_page(lines):
    """Return a list of booleans: is this line part of a letter?"""
    flag = [False] * len(lines)
    i = 0
    while i < len(lines):
        if is_start(lines[i]):
            j = i + 1
            while j < len(lines) and j - i < MAXLEN:
                if is_start(lines[j]) or is_stop(lines[j]):
                    break
                if j + 1 < len(lines) and len(lines[j]) <= 2 and len(lines[j + 1]) <= 2 and j - i >= 3:
                    j += 2
                    while j < len(lines) and len(lines[j]) <= 2:   # rest of signature block
                        j += 1
                    break
                j += 1
            for k in range(i, j):
                flag[k] = True
            i = j
        else:
            i += 1
    return flag


if __name__ == "__main__":
    pages = load_pages()
    rows, samples = [], []
    for _, r in pages.iterrows():
        flag = split_page(r.sents)
        letter = [s for s, f in zip(r.sents, flag) if f]
        other = [s for s, f in zip(r.sents, flag) if not f]
        rows.append({"paper": r.paper, "date": r.date, "page": r.page,
                     "letter_tok": sum(map(len, letter)), "other_tok": sum(map(len, other)),
                     "n_letters": sum(1 for k, f in enumerate(flag) if f and (k == 0 or not flag[k - 1]))})
        k = 0
        while k < len(flag):
            if flag[k]:
                e = k
                while e < len(flag) and flag[e]:
                    e += 1
                samples.append((r.paper, r.date, r.page, r.sents[max(0, k - 1)], r.sents[k:e]))
                k = e
            else:
                k += 1
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letters_segments.csv", index=False)
    print(df.groupby("paper")[["letter_tok", "other_tok", "n_letters"]].sum())
    import random
    random.seed(3)
    pick = random.sample(samples, 30)
    with open(OUT / "letters_sample.md", "w") as fh:
        for p, d, pg, title, body in pick:
            fh.write(f"## {p} {d} p{pg}  ({sum(map(len, body))} tokens, {len(body)} lines)\n")
            fh.write("title?: " + " ".join(title) + "\n\nSTART: " + " ".join(body[0])[:300] + "\n\n")
            fh.write("END: " + " | ".join(" ".join(x) for x in body[-4:])[:300] + "\n\n")
    print(len(samples), "letters")
