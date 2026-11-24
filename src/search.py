"""Command line search. Run: python -m src.search "robot swarm control" """

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.brute_force import search
from src.embeddings import build_index, load_docs, vectorize
from src.hnsw import HNSW

HERE = Path(__file__).resolve().parent.parent
HNSW_PATH = HERE / "data" / "hnsw.npz"


def load_or_build():
    """Load the saved index, or build it if it is not there yet."""
    vectors_path = HERE / "data" / "vectors.npy"
    if not vectors_path.exists():
        print("Building index from data/arxiv_1000.jsonl ...")
        build_index(HERE / "data" / "arxiv_1000.jsonl", HERE / "data")
    docs = load_docs(HERE / "data" / "arxiv_1000.jsonl")
    matrix = np.load(vectors_path)
    vocab = json.loads((HERE / "data" / "vocab.json").read_text())
    idf = np.load(HERE / "data" / "idf.npy")
    return docs, matrix, vocab, idf


def load_or_build_hnsw(matrix, rebuild=False):
    """Load the saved HNSW graph, or build and save it once."""
    if HNSW_PATH.exists() and not rebuild:
        start = time.perf_counter()
        index = HNSW.load(HNSW_PATH)
        print(f"Loaded saved HNSW index in {(time.perf_counter() - start) * 1000:.0f} ms.")
        return index
    print("Building HNSW index, this takes about a minute ...")
    start = time.perf_counter()
    index = HNSW(dim=matrix.shape[1], M=16, ef_construction=100, seed=42)
    index.add_batch(matrix)
    index.save(HNSW_PATH)
    print(f"Built HNSW index in {time.perf_counter() - start:.1f} s, saved to data/hnsw.npz.")
    return index


def main():
    parser = argparse.ArgumentParser(description="Search 1,000 arXiv abstracts")
    parser.add_argument("query", help="search query, in quotes")
    parser.add_argument("--k", type=int, default=10, help="results to show")
    parser.add_argument("--method", choices=["brute", "hnsw"], default="brute",
                        help="brute force (exact) or HNSW (approximate)")
    parser.add_argument("--ef", type=int, default=50, help="HNSW beam width")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the HNSW index instead of loading the saved one")
    args = parser.parse_args()

    docs, matrix, vocab, idf = load_or_build()
    query_vector = vectorize([args.query], vocab, idf)[0]
    if not np.any(query_vector):
        print("No words from your query are in the index, so nothing matches.")
        return

    start = time.perf_counter()
    if args.method == "hnsw":
        index = load_or_build_hnsw(matrix, rebuild=args.rebuild)
        ids, scores = index.search(query_vector, k=args.k, ef=args.ef)
        results = [(float(s), docs[i]) for s, i in zip(scores, ids)]
        label = f"HNSW (ef={args.ef})"
    else:
        results = search(query_vector, matrix, docs, k=args.k)
        label = "Brute force"
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"\nQuery: {args.query}")
    print(f"{label} over {len(docs)} docs: {elapsed_ms:.2f} ms\n")
    for rank, (score, doc) in enumerate(results, 1):
        print(f"{rank}. [{score:.3f}] {doc['title']}")
        print(f"   {doc['id']}  ({doc.get('published', '')})")


if __name__ == "__main__":
    main()
