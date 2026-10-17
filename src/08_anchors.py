"""Fix the anchor word lists for the directional framing test BEFORE running it.

Anchors are derived from text OUTSIDE the test window (Isigidimi 1876-83 + Imvo 1889-94) so they
cannot be tuned to the result:
  religious anchors : words with the highest PMI with the seeds 'tixo', 'kristu'
  political anchors : words with the highest PMI with the seeds 'palamente', 'voti'
PMI is computed from co-occurrence within +-10 tokens inside a sentence. The seeds themselves
(and any word containing a seed stem) are excluded from the anchor sets, and the seed concepts
are excluded from the targets of the directional test.
The lists are exploratory (the religious seeds came from the classifier's top features) and are
meant to be reviewed by a first-language speaker; the file is written once and then fixed.
"""
import json
from collections import Counter

import numpy as np

from common import OUT, load_pages

SEEDS = {"religious": ["tixo", "kristu"], "political": ["palamente", "voti"]}
K, WIN, MIN_C, MIN_CO = 15, 10, 50, 8
MASTHEAD = {"isigidimi", "sigidimi", "samaxosa", "xosa", "xhosa", "imvo", "mvo", "zabantsundu", "ndu",
            "ontsundu", "lovedale", "ngolwesi", "qonce", "komani", "bhayi", "town", "sir", "mr"}


def main():
    pages = load_pages(window=None)
    d = pages.date.str[:4].astype(int)
    held = pages[((pages.paper == "ISIG") & d.between(1876, 1883)) | ((pages.paper == "IMVO") & d.between(1889, 1894))]
    sents = [s for ss in held.sents for s in ss]
    c = Counter(w for s in sents for w in s)
    N = sum(c.values())
    out = {"derived_from": "Isigidimi 1876-83 + Imvo 1889-94", "seeds": SEEDS, "tokens": N}
    for name, seeds in SEEDS.items():
        is_seed = lambda w: any(sd in w for sd in seeds)
        co, n_seed = Counter(), 0
        for s in sents:
            pos = [i for i, w in enumerate(s) if is_seed(w)]
            n_seed += len(pos)
            if not pos:
                continue
            seen = set()
            for i in pos:
                for j in range(max(0, i - WIN), min(len(s), i + WIN + 1)):
                    if j != i and (j not in seen) and not is_seed(s[j]):
                        seen.add(j); co[s[j]] += 1
        pmi = {w: np.log(co[w] * N / (c[w] * n_seed)) for w in co
               if c[w] >= MIN_C and co[w] >= MIN_CO and len(w) > 2 and w not in MASTHEAD}
        out[name] = [w for w, _ in sorted(pmi.items(), key=lambda kv: -kv[1])[:K]]
        out[name + "_pmi"] = {w: round(float(pmi[w]), 2) for w in out[name]}
        print(name, "seed tokens:", n_seed, "\n ", out[name])
    (OUT / "anchors.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
