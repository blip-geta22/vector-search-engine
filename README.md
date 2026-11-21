# Vector Search Engine from scratch

Search 1,000 research papers by meaning-ish, not just keywords.
Built in Python and NumPy only. No FAISS, no vector database.

**Status: Step 1 done.** Embeddings plus a brute force baseline work
and are tested. HNSW, the CLI persistence beyond this step, and the
benchmark are not built yet. They are Steps 2 to 4.

## What Step 1 does

Each paper abstract becomes a TF-IDF vector: a word counts more if
it is common in that paper and rare across all papers. Vectors are
normalised to length 1, so cosine similarity is a dot product.
Brute force search compares a query to all 1,000 papers and returns
the top 10. That is the ground truth every faster index is measured
against later.

TF-IDF is a deliberate Step 1 choice, not the final embedding:
it matches words, not meaning, so a query like "transformer language
models" returns papers that share those words, with low scores.
A neural embedding model can replace `src/embeddings.py` later
without changing the search code.

## Corpus

1,000 real arXiv abstracts, fetched 20 Nov 2026 from the arXiv API
(`cat:cs.AI OR cat:cs.LG OR cat:cs.CL`, most recent first),
in `data/arxiv_1000.jsonl`. One JSON object per line: id, title,
summary, published date.

## How to run it

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.search "robot swarm control"
```

The first search builds the index and saves it to `data/`; later
searches just load it. The generated `vectors.npy`, `idf.npy` and
`vocab.json` are gitignored and rebuilt by running the search.

## Measured, 20 Nov 2026, this machine

- Corpus: 1,000 docs, vocabulary 5,000 terms, matrix shape (1000, 5000)
- Tests: 3 passed (`python -m unittest discover -s tests -v`)
- Brute force search, 1,000 docs: about 5.6 ms per query
  ("robot swarm control": 5.65 ms, "transformer language models": 5.59 ms)
- Sample top result, "robot swarm control": "MAWAM: A Multi-Agent
  World-Action Model for Coordinated Manipulation", score 0.267

No HNSW numbers exist yet. No recall, speedup, or 10,000 / 40,000
doc numbers exist yet. Those come from the benchmark in Step 4,
and only measured numbers go in this README.

## Files

- `src/embeddings.py` — tokenize, TF-IDF, build and save the index
- `src/brute_force.py` — cosine search over every document
- `src/search.py` — command line search, top 10 with scores
- `tests/test_brute_force.py` — 3 tests, standard library only
- `data/arxiv_1000.jsonl` — the corpus (1.6 MB)
