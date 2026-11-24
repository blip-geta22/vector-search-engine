# Vector Search Engine from scratch

Search 1,000 research papers by similarity, not just keywords.
Python and NumPy only. No FAISS, no vector database.

Each abstract becomes a TF-IDF vector. A word counts more if it is common in that paper and rare across all papers. Vectors are normalised to length 1, so similarity is a dot product.

Brute force compares a query to all 1,000 papers and returns the top 10. HNSW is a graph index built from scratch that returns almost the same top 10, faster at low ef.

TF-IDF matches words, not meaning, so "transformer language models" returns papers that share those words, with low scores. A neural embedding could replace `src/embeddings.py` later without changing the search code.

## Data

1,000 real arXiv abstracts, fetched 20 Nov 2026 from the arXiv API (cs.AI, cs.LG, cs.CL, most recent first), in `data/arxiv_1000.jsonl`. One JSON object per line: id, title, summary, published date.

## How to run it

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.search "robot swarm control"
python -m src.search "robot swarm control" --method hnsw --ef 10
python -m benchmarks.benchmark
```

The first HNSW search builds the index (about a minute) and saves it to `data/hnsw.npz`. Later searches just load it, in about 50 ms. Use `--rebuild` to build it again. The generated `vectors.npy`, `idf.npy`, `vocab.json` and `hnsw.npz` are gitignored.

## Measured, 23 Nov 2026

Corpus: 1,000 docs, vocabulary 5,000 terms, matrix shape (1000, 5000). Queries: 20 random docs, seed 42. Numbers from a 1-vCPU sandbox, they vary about 10% run to run.

- Tests: 12 passed
- Brute force search: about 5.5 ms per query
- HNSW build: 53.6 s, saved index 20.1 MB, load about 50 ms
- HNSW, ef=10: recall@10 0.98, about 2.1 ms per query, about 2.6x faster than brute force
- HNSW, ef=50: recall@10 1.00, about 5.4 ms per query, about the same speed as brute force
- HNSW, ef=100: recall@10 1.00, about 7.3 ms per query, slower than brute force
- Sample top result, "robot swarm control": "MAWAM: A Multi-Agent World-Action Model for Coordinated Manipulation", score 0.267, same top 3 by brute force and by HNSW at ef=10

At 1,000 docs HNSW only wins at low ef. That is expected: brute force is one fast matrix multiply, HNSW pays Python overhead per hop. The win should grow with a bigger corpus.

## Files

- `src/embeddings.py` — tokenize, TF-IDF, build and save the index
- `src/brute_force.py` — search over every document
- `src/hnsw.py` — HNSW graph, plus save and load
- `src/search.py` — command line search, top 10 with scores
- `benchmarks/benchmark.py` — brute force vs HNSW, recall and latency
- `tests/test_brute_force.py` — 6 tests
- `tests/test_hnsw.py` — 6 tests, including recall against brute force
- `data/arxiv_1000.jsonl` — the corpus (1.6 MB)
