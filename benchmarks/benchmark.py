"""Benchmark brute force vs HNSW on held-out queries.

Split: 800 docs go in the index, the other 200 are queries only.
They were never indexed, so a query cannot just find itself.
Seed 42, so the split and the numbers repeat.
"""

import tempfile
import time
from pathlib import Path

import numpy as np

from src.embeddings import load_docs
from src.hnsw import HNSW
from src.search import load_or_build

HERE = Path(__file__).resolve().parent.parent


def main():
    docs, matrix, vocab, idf = load_or_build()
    rng = np.random.default_rng(42)
    order = rng.permutation(len(docs))
    index_idx, query_idx = order[:800], order[800:]
    index_matrix = matrix[index_idx]
    queries = matrix[query_idx]
    n = len(queries)

    # Brute force: mean time and top 10 positions for each query.
    start = time.perf_counter()
    brute_ids = []
    for q in queries:
        scores = index_matrix @ q
        top = np.argpartition(-scores, 9)[:10]
        brute_ids.append(set(top.tolist()))
    brute_ms = (time.perf_counter() - start) * 1000 / n

    # HNSW on the 800-doc index only, timed build, size from a temp save.
    start = time.perf_counter()
    index = HNSW(dim=matrix.shape[1], M=16, ef_construction=100, seed=42)
    index.add_batch(index_matrix)
    build_s = time.perf_counter() - start
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "hnsw800.npz"
        index.save(path)
        size_mb = path.stat().st_size / 1e6

    print(f"Docs indexed: 800, held-out queries: {n}, vocab: {matrix.shape[1]}")
    print(f"HNSW build: {build_s:.1f} s, saved index: {size_mb:.1f} MB")
    print(f"Brute force: {brute_ms:.2f} ms/query")
    print()
    print("ef    recall@10  hnsw ms/query  speedup")
    for ef in (10, 50, 100):
        recalls = []
        start = time.perf_counter()
        for q, truth in zip(queries, brute_ids):
            ids, _ = index.search(q, k=10, ef=ef)
            recalls.append(len(set(ids.tolist()) & truth) / 10)
        hnsw_ms = (time.perf_counter() - start) * 1000 / n
        print(f"{ef:<6}{np.mean(recalls):.3f}      {hnsw_ms:.2f}           {brute_ms / hnsw_ms:.2f}x")


if __name__ == "__main__":
    main()
