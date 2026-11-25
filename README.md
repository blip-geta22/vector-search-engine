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

The first HNSW search builds the index and saves it to `data/hnsw.npz`. Later searches just load it, in about 50 ms. Use `--rebuild` to build it again. The generated `vectors.npy`, `idf.npy`, `vocab.json` and `hnsw.npz` are gitignored.

## Measured, 24 Nov 2026

Corpus: 1,000 docs, vocabulary 5,000 terms, matrix shape (1000, 5000). The benchmark splits it: 800 docs go in the index, the other 200 are queries only, so a query cannot just find itself. Seed 42, so the split repeats. Run `python -m benchmarks.benchmark` to repeat it.

Hardware: AMD EPYC 9D25, 4 vCPUs in a shared Linux sandbox. Absolute latencies and speedups are specific to this hardware and do not travel to other machines. On a faster machine, the same 800-doc held-out benchmark measured brute force 0.70 ms per query and HNSW ef=10 0.64 ms, only 1.10x faster, and ef=50 only 0.48x as fast as brute force. Only the direction travels: HNSW wins only at low ef. Recall was identical across three runs here; latencies varied about 10% run to run.

- Tests: 12 passed
- Brute force search: about 4.4 ms per query (n=200 held-out queries)
- HNSW build, 800 docs: 41.4 s, saved index 16.1 MB
- HNSW, ef=10: recall@10 0.81 (n=200), about 2.1 ms per query, about 2.1x faster than brute force on this hardware
- HNSW, ef=50: recall@10 0.99 (n=200), about 5.0 ms per query, about the same speed as brute force on this hardware
- HNSW, ef=100: recall@10 1.00 (n=200), about 6.5 ms per query, slower than brute force on this hardware
- Full 1,000-doc index: build 53.0 s, saved index 20.1 MB, load 25 to 130 ms
- Sample top result, "robot swarm control": "MAWAM: A Multi-Agent World-Action Model for Coordinated Manipulation", score 0.267, same top 3 by brute force and by HNSW at ef=10

Recall here means how well HNSW copies brute force, not search quality. At 1,000 docs HNSW only wins at low ef, and at ef=10 it misses about one in five of the brute force top 10. That is expected: brute force is one fast matrix multiply, HNSW pays Python overhead per hop.

## Scaling, 24 Nov 2026

Run `python -m benchmarks.benchmark_scaling` to repeat it. Corpus: 159,749 real arXiv abstracts, TF-IDF 5,000 dims, seed 42. The vocabulary was fitted once, on the 10,000 indexed docs. Both sizes use the same 200 held-out queries, so they are comparable. The 1,000-doc numbers here differ slightly from the main benchmark above because that one indexes 800 docs with a vocabulary fitted on 1,000 docs, while this one indexes a full 1,000 docs with a vocabulary fitted on 10,000. Hardware: AMD EPYC 9D25, 4 vCPUs, same sandbox as above, so these speedups are specific to this hardware too.

- 1,000 docs (n=200): brute force 6.04 ms per query, HNSW build 53.7 s, saved index 20.1 MB. ef=10: recall@10 0.864, 2.17 ms, 2.78x faster. ef=50: recall@10 0.990, 5.34 ms, 1.13x faster. ef=100: recall@10 0.999, 7.37 ms, 0.82x, slower.
- 10,000 docs (n=200): brute force 54.12 ms per query, HNSW build 667.0 s, saved index 201.2 MB. ef=10: recall@10 0.810, 3.80 ms, 14.25x faster. ef=50: recall@10 0.959, 9.24 ms, 5.85x faster. ef=100: recall@10 0.985, 14.92 ms, 3.63x faster.

At 10,000 docs the win is real: brute force grows with the corpus, HNSW barely does, so ef=10 is about 14x faster at recall 0.81. The cost is the build, 667 seconds for 10,000 docs. 40,000 docs was not run: at that build rate it would take well over 15 minutes here, so the scaling run stops at 10,000.

## Files

- `src/embeddings.py`: tokenize, TF-IDF, build and save the index
- `src/brute_force.py`: search over every document
- `src/hnsw.py`: HNSW graph, plus save and load
- `src/search.py`: command line search, top 10 with scores
- `benchmarks/benchmark.py`: brute force vs HNSW, recall and latency, held-out queries
- `benchmarks/benchmark_scaling.py`: scaling run at 1,000 and 10,000 docs, held-out queries, TF-IDF vectors
- `benchmarks/scaling_results.json`: measured scaling numbers
- `tests/test_brute_force.py`: 6 tests
- `tests/test_hnsw.py`: 6 tests, including recall against brute force
- `data/arxiv_1000.jsonl`: the corpus (1.6 MB)
