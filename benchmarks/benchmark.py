"""Benchmark brute force vs HNSW on the real 1,000-doc index."""

import time
from pathlib import Path

import numpy as np

from src.brute_force import search as brute_search
from src.embeddings import load_docs
from src.hnsw import HNSW
from src.search import HNSW_PATH, load_or_build, load_or_build_hnsw

HERE = Path(__file__).resolve().parent.parent


def main():
    docs, matrix, vocab, idf = load_or_build()
    # 20 random docs as queries, fixed seed so the numbers repeat.
    rng = np.random.default_rng(42)
    queries = matrix[rng.integers(0, len(docs), 20)]
    n = len(queries)

    # Brute force: mean time and top 10 positions for each query.
    start = time.perf_counter()
    brute_ids = []
    for q in queries:
        scores = matrix @ q
        top = np.argpartition(-scores, 9)[:10]
        brute_ids.append(set(top.tolist()))
    brute_ms = (time.perf_counter() - start) * 1000 / n

    index = load_or_build_hnsw(matrix)
    print(f"Docs: {len(docs)}, vocab: {matrix.shape[1]}")
    print(f"Index file: {HNSW_PATH.name}, {HNSW_PATH.stat().st_size / 1e6:.1f} MB")
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
