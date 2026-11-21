"""Brute force search: compare a query to every document.

This is the simple, correct way to search. It is slow when the
corpus is large, because it checks all 1,000 papers for every
query, but it is the ground truth: every faster index we build
later (HNSW in Step 2) is measured against this one.

Vectors are normalised to length 1, so cosine similarity is
just the dot product of the query vector and each document row.
"""

import numpy as np


def search(query_vector, matrix, docs, k=10):
    """Return the top k docs as (score, doc) pairs, best first."""
    scores = matrix @ query_vector
    # argpartition is faster than a full sort; then sort just the top k.
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [(float(scores[i]), docs[i]) for i in top]
