"""Tests for HNSW."""

import unittest

import numpy as np

from src.brute_force import search as brute_force_search
from src.hnsw import HNSW


def clustered_vectors(n, dim, n_clusters=5, seed=0):
    """Unit vectors drawn around random cluster centres."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_clusters, dim))
    x = centres[rng.integers(0, n_clusters, n)] + 0.5 * rng.normal(size=(n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def make_index(n=300, dim=32, seed=7):
    x = clustered_vectors(n, dim, seed=seed)
    h = HNSW(dim, M=12, ef_construction=100, seed=seed)
    h.add_batch(x)
    return h, x


class TestHNSW(unittest.TestCase):
    def test_add_returns_sequential_ids(self):
        h = HNSW(8, M=4, ef_construction=20)
        x = clustered_vectors(50, 8, n_clusters=3)
        ids = [h.add(v) for v in x]
        self.assertEqual(ids, list(range(50)))
        self.assertEqual(len(h), 50)

    def test_empty_index_returns_nothing(self):
        h = HNSW(8)
        ids, scores = h.search(np.ones(8), k=10)
        self.assertEqual(len(ids), 0)

    def test_search_finds_inserted_vector(self):
        h, x = make_index()
        ids, scores = h.search(x[0], k=5, ef=50)
        self.assertEqual(ids[0], 0)
        self.assertAlmostEqual(scores[0], 1.0, places=5)

    def test_results_sorted_and_unique(self):
        h, x = make_index()
        ids, scores = h.search(x[10], k=10, ef=50)
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(set(ids.tolist())), 10)
        self.assertTrue(np.all(np.diff(scores) <= 1e-6))

    def test_recall_against_brute_force(self):
        h, x = make_index()
        docs = list(range(len(x)))
        recalls = []
        for q in x[:20]:
            ids, _ = h.search(q, k=10, ef=100)
            bf = brute_force_search(q, x, docs, k=10)
            bf_ids = {d for _, d in bf}
            recalls.append(len(set(ids.tolist()) & bf_ids) / 10)
        self.assertGreaterEqual(float(np.mean(recalls)), 0.95)

    def test_save_and_load_roundtrip(self):
        import tempfile
        from pathlib import Path

        h, x = make_index()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "idx.npz"
            h.save(path)
            loaded = HNSW.load(path)
            self.assertEqual(len(loaded), len(h))
            a, _ = h.search(x[0], k=5, ef=50)
            b, _ = loaded.search(x[0], k=5, ef=50)
            self.assertEqual(a.tolist(), b.tolist())


if __name__ == "__main__":
    unittest.main()
