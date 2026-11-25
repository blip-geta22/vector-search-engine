"""Download and prepare the full arXiv corpus used by the benchmark.

Source (nothing here is invented):
  The Kaggle dataset "Cornell-University/arxiv" (arXiv metadata: title + abstract),
  in the cleaned copy published in the GitHub repo
  safakkbilici/Academic-Paper-Title-Recommendation  ->  data/df_to_model.tar.gz
  In that copy punctuation has already been stripped from the abstracts.
  Columns in the original CSV: input_text (= abstract), target_text (= title).

What this script does:
  1. downloads the tar.gz (about 48 MB) and extracts the CSV
  2. collapses whitespace, drops rows with an empty title/abstract and
     exact-duplicate abstracts
  3. writes data/corpus.csv            (title,abstract - full corpus, git-ignored)
  4. writes data/sample_abstracts.csv  (1,000 random docs, small enough to commit)

Usage:  python data/download_corpus.py
"""
import csv
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

URL = ("https://raw.githubusercontent.com/safakkbilici/"
       "Academic-Paper-Title-Recommendation/main/data/df_to_model.tar.gz")
HERE = Path(__file__).resolve().parent
SAMPLE_SIZE = 1000
SEED = 42  # must match SEED in benchmarks/benchmark_scaling.py

csv.field_size_limit(sys.maxsize)


def main():
    print(f"Downloading {URL} ...")
    raw = urllib.request.urlopen(URL, timeout=300).read()
    print(f"  downloaded {len(raw) / 1e6:.1f} MB")

    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(".csv"))
        text = tar.extractfile(member).read().decode("utf-8")

    rows = list(csv.DictReader(io.StringIO(text)))
    print(f"  rows in source CSV: {len(rows):,}")

    seen, docs = set(), []
    for r in rows:
        # arXiv titles are hard-wrapped over several lines; collapse all whitespace.
        abstract = " ".join(r["input_text"].split())
        title = " ".join(r["target_text"].split())
        if not abstract or not title or abstract in seen:
            continue
        seen.add(abstract)
        docs.append((title, abstract))
    print(f"  after dropping empty / duplicate-abstract rows: {len(docs):,}")

    def write(path, items):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["title", "abstract"])
            w.writerows(items)

    write(HERE / "corpus.csv", docs)
    order = np.random.default_rng(SEED).permutation(len(docs))
    write(HERE / "sample_abstracts.csv", [docs[i] for i in order[:SAMPLE_SIZE]])
    words = np.array([len(a.split()) for _, a in docs])
    print(f"  abstract length in words: median {int(np.median(words))}, "
          f"min {words.min()}, max {words.max()}")
    print(f"Wrote {HERE/'corpus.csv'} ({len(docs):,} docs) and "
          f"{HERE/'sample_abstracts.csv'} ({SAMPLE_SIZE} docs)")


if __name__ == "__main__":
    main()
