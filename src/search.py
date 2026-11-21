"""Command line search. Run: python -m src.search "robot swarm control" """

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.brute_force import search
from src.embeddings import build_index, load_docs, vectorize

HERE = Path(__file__).resolve().parent.parent


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


def main():
    parser = argparse.ArgumentParser(description="Search 1,000 arXiv abstracts")
    parser.add_argument("query", help="search query, in quotes")
    parser.add_argument("--k", type=int, default=10, help="results to show")
    args = parser.parse_args()

    docs, matrix, vocab, idf = load_or_build()
    query_vector = vectorize([args.query], vocab, idf)[0]

    start = time.perf_counter()
    results = search(query_vector, matrix, docs, k=args.k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"\nQuery: {args.query}")
    print(f"Brute force over {len(docs)} docs: {elapsed_ms:.2f} ms\n")
    for rank, (score, doc) in enumerate(results, 1):
        print(f"{rank}. [{score:.3f}] {doc['title']}")
        print(f"   {doc['id']}  ({doc['published']})")


if __name__ == "__main__":
    main()
