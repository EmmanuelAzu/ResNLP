"""Share of alphabetic ASCII vocabulary entries containing th/kh/ph, per tokenizer (mechanism check for 05)."""
import csv, re
import importlib.util, sys
sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location("fert", "src/05_fertility.py")
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
asp = re.compile(r"(th|kh|ph)")
rows = []
for name, repo in f.TOKENIZERS:
    t = f.load_tokenizer(repo)
    v = ([t.sp.id_to_piece(i) for i in range(t.sp.get_piece_size())] if hasattr(t, "sp")
         else list(t.get_vocab()))
    v = [x.lstrip("▁#") for x in v]
    v = [x for x in v if x.isalpha() and x.isascii()]
    a = sum(1 for x in v if asp.search(x))
    rows.append((name, len(v), a, 100 * a / len(v)))
with open(f.OUT / "vocab_aspiration.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["tokenizer", "entries", "with_th_kh_ph", "percent"]); w.writerows(rows)
print(rows)
