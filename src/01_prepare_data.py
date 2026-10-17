"""Unzip the isiXhosa page files and build data/pages.csv.gz (cleaned, one row per page).

Usage: python src/01_prepare_data.py
Reads TxtLangFiles1870-1894_TB.zip from the repository root (XhTxt/ = isiXhosa blocks only).
"""
import re
import zipfile

import pandas as pd

from common import DATA, PAGES_CSV, ROOT, clean_to_sentences

ZIP = ROOT / "TxtLangFiles1870-1894_TB.zip"
NAME_RE = re.compile(r"XhTxt/((IMVO|ISIG)_(\d{4}-\d\d-\d\d)_p(\d+))\.txt$")


def main():
    DATA.mkdir(exist_ok=True)
    rows = []
    with zipfile.ZipFile(ZIP) as z:
        for name in sorted(z.namelist()):
            m = NAME_RE.search(name)
            if not m or name.startswith("__MACOSX"):
                continue
            raw = z.read(name).decode("utf-8", errors="replace")
            sents = clean_to_sentences(raw)
            rows.append({"paper": m.group(2), "date": m.group(3), "page": int(m.group(4)),
                         "text": "\n".join(" ".join(s) for s in sents)})
    df = pd.DataFrame(rows)
    df.to_csv(PAGES_CSV, index=False)
    print(f"{len(df)} pages -> {PAGES_CSV}")
    print(df.groupby("paper").size().to_string())


if __name__ == "__main__":
    main()
