# Vector Search Engine from scratch

Search 1,000 research papers by similarity, not just keywords.
Python and NumPy only. No FAISS, no vector database.

Step 1 is done. Embeddings plus a brute force baseline, tested.
HNSW and the benchmark are next.

## How it works

Each abstract becomes a TF-IDF vector. A word counts more if it is common in that paper and rare across all papers. Vectors are normalised to length 1, so similarity is a dot product.

Brute force compares a query to all 1,000 papers and returns the top 10. That is the baseline HNSW will be measured against later.

TF-IDF matches words, not meaning, so "transformer language models" returns papers that share those words, with low scores. A neural embedding could replace `src/embeddings.py` later without changing the search code.

## Data

1,000 real arXiv abstracts, fetched 20 Nov 2026 from the arXiv API (cs.AI, cs.LG, cs.CL, most recent first), in `data/arxiv_1000.jsonl`. One JSON object per line: id, title, summary, published date.

## How to run it

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.search "robot swarm control"
```

The first search builds the index and saves it to `data/`. Later searches just load it. The generated `vectors.npy`, `idf.npy` and `vocab.json` are gitignored.

## Measured, 20 Nov 2026

- Corpus: 1,000 docs, vocabulary 5,000 terms, matrix shape (1000, 5000)
- Tests: 5 passed
- Brute force search: about 5.6 ms per query
- Sample top result, "robot swarm control": "MAWAM: A Multi-Agent World-Action Model for Coordinated Manipulation", score 0.267

No HNSW numbers yet. Those come with the benchmark.

## Files

- `src/embeddings.py` — tokenize, TF-IDF, build and save the index
- `src/brute_force.py` — search over every document
- `src/search.py` — command line search, top 10 with scores
- `tests/test_brute_force.py` — 5 tests
- `data/arxiv_1000.jsonl` — the corpus (1.6 MB)
