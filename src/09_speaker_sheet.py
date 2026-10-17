"""Build the one-hour sheet for a first-language isiXhosa speaker (outputs/speaker_sheet.md).

Sections: (1) the aspiration-stripping rules with examples, (2) the stems and the surface forms
they match, (3) the anchor lists, (4) a BLIND neighbour comparison (which paper is X or Y is
randomised; the key is written to outputs/speaker_key.json and should not be shown), (5) abantu.
"""
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from common import CONTROL, EXPLORATORY, OUT, THEORY, DATA, load_pages
import importlib.util

spec = importlib.util.spec_from_file_location("fert", str(OUT.parent / "src" / "05_fertility.py"))
fert = importlib.util.module_from_spec(spec); spec.loader.exec_module(fert)

rng = np.random.default_rng(2026)
pages = load_pages(window=None)
hist_counts = Counter(w for ss in pages.sents for s in ss for w in s)
win = load_pages()
win_counts = Counter(w for ss in win.sents for s in ss for w in s)
lines = ["# Speaker check sheet", "",
         "Thank you for helping. These are historical (1870s-1880s) newspaper texts in the old orthography. "
         "Please mark anything that is wrong or doubtful; short notes are fine.", ""]

# 1. rules
lines += ["## 1. Spelling rules (modern -> old spelling)", "",
          "We rewrite modern spelling to approximate the old one by these rules: th->t, kh->k, ph->p, rh->r, ch->c, qh->q, xh->x. "
          "We deliberately leave sh, hl, tsh, bh, gq, dl, nh, mh etc. unchanged. Are the rules right? What is missing or wrong?", "",
          "| rule | modern word (in news text) | rewritten | rewritten form occurs in the old newspapers (count) |", "|---|---|---|---|"]
modern = fert.modern_sentences()
mc = Counter(w for s in modern for w in s)
for a, b in fert.STRIP:
    cand = [w for w, n in mc.most_common(6000) if a in w and len(w) > 4]
    seen_ = [w for w in cand if hist_counts.get(fert.strip_aspiration(w), 0) > 0][:3]
    ex = seen_ + [w for w in cand if w not in seen_ and hist_counts.get(fert.strip_aspiration(w), 0) == 0][:2]
    for w in ex:
        r = fert.strip_aspiration(w)
        lines.append(f"| {a}->{b} | {w} | {r} | {hist_counts.get(r, 0)} |")
lines += [""]

# 2. stems
lines += ["## 2. Concept stems and the word forms they match (Nov 1884-1888)", "",
          "For each concept, are these forms really about that concept? Please cross out wrong ones and add forms we missed.", ""]
for grp, table in (("political", THEORY), ("control", CONTROL), ("religion (exploratory)", EXPLORATORY)):
    for name, stems in table.items():
        forms = sorted({w: c for w, c in win_counts.items() if any(s in w for s in stems) and c >= 5}.items(), key=lambda kv: -kv[1])[:14]
        lines.append(f"- **{grp}: {name}** (stems: {', '.join(stems)}): " + ", ".join(f"{w} ({c})" for w, c in forms))
lines += [""]

# 3. anchors
anc = json.loads((OUT / "anchors.json").read_text())
lines += ["## 3. Anchor words", "",
          "These were picked by a statistical procedure (words that co-occur with God/Christ, or with parliament/vote), not by hand. "
          "Which belong to a religious or a political set and which do not?", "",
          "- Religious: " + ", ".join(anc["religious"]), "- Political: " + ", ".join(anc["political"]), ""]

# 4/5. blind neighbours
nb = pd.read_csv(OUT / "neighbours_stable.csv")
words = [w for w in nb.word.unique()]
key = {}
lines += ["## 4. Blind neighbour comparison", "",
          "For each word, two lists of words that appear in similar contexts, from two different newspapers (X and Y). "
          "Please say: (a) are the lists similar in meaning, or does one lean towards a different theme "
          "(religious / political / everyday / spelling variants / nonsense)? (b) any word that does not belong?", ""]
focus = ["abantu", "umntu", "abantsundu", "abamnyama"] + [w for w in words if w not in ("abantu", "umntu", "abantsundu", "abamnyama")]
for w in focus:
    sub = nb[nb.word == w]
    if sub.empty:
        continue
    flip = bool(rng.integers(2))
    key[w] = {"X": "Imvo" if flip else "Isigidimi", "Y": "Isigidimi" if flip else "Imvo"}
    lists = {p: ", ".join(sub[sub.paper == p].sort_values("in_models", ascending=False).neighbour.head(8)) for p in ("Isigidimi", "Imvo")}
    x, y = (lists["Imvo"], lists["Isigidimi"]) if flip else (lists["Isigidimi"], lists["Imvo"])
    lines += [f"**{w}**", f"- X: {x}", f"- Y: {y}", ""]
ms = pd.read_csv(OUT / "most_shifted_words.csv")
lines += ["## 5. Words whose company differs most between the two papers", "",
          "A statistical procedure flagged these. For each: what does it mean, and is the difference about content, "
          "grammar/spelling convention (e.g. written as one word in one paper and two in the other), or OCR damage?", "",
          "| word | uses in Isigidimi | uses in Imvo (same size sample) |", "|---|---|---|"]
for r in ms.itertuples():
    lines.append(f"| {r.word} | {int(r.n_isig)} | {r.n_imvo_draw:.0f} |")
lines += [""]
(OUT / "speaker_key.json").write_text(json.dumps(key, indent=1))
(OUT / "speaker_sheet.md").write_text("\n".join(lines))
print("wrote speaker_sheet.md with", len(key), "blind word pairs")
