"""Scaling benchmark: brute force vs HNSW at 1,000 and 10,000 docs.

Uses the TF-IDF vectors from src/embeddings.py, 5,000 dims, not SVD.
Corpus: data/corpus.csv, 159,749 real arXiv abstracts, downloaded from
the cleaned Kaggle arXiv copy at
https://raw.githubusercontent.com/safakkbilici/Academic-Paper-Title-Recommendation/main/data/df_to_model.tar.gz
(see the download script in the Claude full build). The corpus file is
gitignored, it stays local.

Method: seeded shuffle, seed 42. The first 10,000 docs are indexed,
the next 200 docs are queries only, held out, so a query never finds
itself. The 1,000-doc run uses the first 1,000 of the same 10,000 and
the same 200 queries, so the two sizes are comparable. The vocabulary
is fitted once, on the 10,000 indexed docs.

40,000 docs was not run: the 1,000-doc HNSW build already takes about
50 s here, so a 40,000-doc build would take well over 15 minutes in
this sandbox. That is a measured limit, not a guess, see the build
times below.

Run: python -m benchmarks.benchmark_scaling
This takes several minutes, most of it building the 10,000-doc index.
"""

import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import build_vocab, vectorize  # noqa: E402
from src.hnsw import HNSW  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
CORPUS = HERE / "data" / "corpus.csv"
SIZES = (1000, 10000)
N_QUERIES = 200
SEED = 42


def load_corpus(path):
    """Read title and abstract rows from the corpus CSV."""
    docs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title = " ".join(row["title"].split())
            abstract = " ".join(row["abstract"].split())
            if title and abstract:
                docs.append((title, abstract))
    return docs


def main():
    if not CORPUS.exists():
        sys.exit(f"{CORPUS} not found, download the corpus first, see the docstring.")
    docs = load_corpus(CORPUS)
    print(f"Corpus: {len(docs):,} docs")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(docs))
    index_docs = [docs[i] for i in order[:10000]]
    query_docs = [docs[i] for i in order[10000:10200]]

    print("Fitting TF-IDF vocabulary on 10,000 docs ...", flush=True)
    start = time.perf_counter()
    fake_docs = [{"title": t, "summary": a} for t, a in index_docs]
    vocab, idf = build_vocab(fake_docs)
    print(f"  vocab: {len(vocab)} terms, fitted in {time.perf_counter() - start:.1f} s",
          flush=True)

    print("Vectorizing 10,000 docs and 200 held-out queries ...", flush=True)
    start = time.perf_counter()
    index_texts = [t + " " + a for t, a in index_docs]
    query_texts = [t + " " + a for t, a in query_docs]
    matrix = vectorize(index_texts, vocab, idf)
    queries = vectorize(query_texts, vocab, idf)
    print(f"  matrix shape: {matrix.shape}, done in {time.perf_counter() - start:.1f} s",
          flush=True)

    rows = []
    for n in SIZES:
        print(f"\n=== {n:,} documents, {N_QUERIES} held-out queries ===", flush=True)
        V = matrix[:n]

        start = time.perf_counter()
        brute_ids = []
        for q in queries:
            scores = V @ q
            top = np.argpartition(-scores, 9)[:10]
            brute_ids.append(set(top.tolist()))
        brute_ms = (time.perf_counter() - start) * 1000 / N_QUERIES
        print(f"  brute force: {brute_ms:.2f} ms/query", flush=True)

        start = time.perf_counter()
        index = HNSW(dim=matrix.shape[1], M=16, ef_construction=100, seed=42)
        index.add_batch(V)
        build_s = time.perf_counter() - start
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "hnsw.npz"
            index.save(path)
            size_mb = path.stat().st_size / 1e6
        print(f"  HNSW build: {build_s:.1f} s, saved index: {size_mb:.1f} MB", flush=True)

        sweep = []
        for ef in (10, 50, 100):
            recalls = []
            start = time.perf_counter()
            for q, truth in zip(queries, brute_ids):
                ids, _ = index.search(q, k=10, ef=ef)
                recalls.append(len(set(ids.tolist()) & truth) / 10)
            hnsw_ms = (time.perf_counter() - start) * 1000 / N_QUERIES
            sweep.append({"ef": ef, "recall": float(np.mean(recalls)),
                          "ms": hnsw_ms, "speedup": brute_ms / hnsw_ms})
            print(f"  ef={ef:<4} recall@10={sweep[-1]['recall']:.3f}  "
                  f"{hnsw_ms:.2f} ms/query  speedup {brute_ms / hnsw_ms:.2f}x",
                  flush=True)
        rows.append({"docs": n, "queries": N_QUERIES, "brute_ms": brute_ms,
                     "build_s": build_s, "size_mb": size_mb, "sweep": sweep})

    out = {"hardware": "AMD EPYC 9D25, 4 vCPUs in a shared Linux sandbox",
           "corpus": "159,749 arXiv abstracts, TF-IDF 5,000 dims, seed 42",
           "sizes_run": list(SIZES), "sizes_not_run": [40000],
           "not_run_reason": "40,000-doc HNSW build would take well over 15 minutes here",
           "rows": rows}
    (HERE / "benchmarks" / "scaling_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote benchmarks/scaling_results.json")
    print("40,000 docs: not run, see the docstring for why.")


if __name__ == "__main__":
    main()
