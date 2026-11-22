"""Tests for Step 1."""

import unittest

import numpy as np

from src.brute_force import search
from src.embeddings import build_vocab, tokenize, vectorize


class TestEmbeddings(unittest.TestCase):
    def test_tokenize_drops_stopwords(self):
        self.assertEqual(tokenize("The cat and the dog"), ["cat", "dog"])

    def test_vectors_are_unit_length(self):
        docs = [
            {"title": "cats", "summary": "cats sleep all day"},
            {"title": "dogs", "summary": "dogs run in parks"},
        ]
        vocab, idf = build_vocab(docs, max_terms=100, min_docs=1)
        matrix = vectorize(["cats sleep", "dogs run"], vocab, idf)
        lengths = np.linalg.norm(matrix, axis=1)
        np.testing.assert_allclose(lengths, [1.0, 1.0], atol=1e-5)

    def test_search_empty_docs_returns_empty(self):
        matrix = np.zeros((0, 3))
        query = np.array([1.0, 0.0, 0.0])
        self.assertEqual(search(query, matrix, [], k=10), [])

    def test_search_clamps_k_to_doc_count(self):
        docs = [
            {"title": "cats", "summary": "cats sleep all day"},
            {"title": "dogs", "summary": "dogs run in parks"},
        ]
        vocab, idf = build_vocab(docs, max_terms=100, min_docs=1)
        matrix = vectorize(
            [d["title"] + " " + d["summary"] for d in docs], vocab, idf
        )
        query = vectorize(["cats sleep"], vocab, idf)[0]
        results = search(query, matrix, docs, k=10)
        self.assertEqual(len(results), 2)

    def test_search_returns_k_in_order(self):
        docs = [
            {"title": "cats", "summary": "cats sleep all day"},
            {"title": "dogs", "summary": "dogs run in parks"},
            {"title": "birds", "summary": "birds fly in trees"},
        ]
        vocab, idf = build_vocab(docs, max_terms=100, min_docs=1)
        matrix = vectorize(
            [d["title"] + " " + d["summary"] for d in docs], vocab, idf
        )
        query = vectorize(["cats sleep"], vocab, idf)[0]
        results = search(query, matrix, docs, k=2)
        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0][0], results[1][0])

    def test_same_words_score_higher(self):
        docs = [
            {"title": "cats", "summary": "cats sleep all day"},
            {"title": "dogs", "summary": "dogs run in parks"},
        ]
        vocab, idf = build_vocab(docs, max_terms=100, min_docs=1)
        matrix = vectorize(
            [d["title"] + " " + d["summary"] for d in docs], vocab, idf
        )
        query = vectorize(["cats sleep"], vocab, idf)[0]
        results = search(query, matrix, docs, k=2)
        self.assertEqual(results[0][1]["title"], "cats")
        self.assertGreater(results[0][0], results[1][0])


if __name__ == "__main__":
    unittest.main()
