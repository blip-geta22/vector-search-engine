"""Brute force search. Checks every doc, slow but always correct."""

import numpy as np


def search(query_vector, matrix, docs, k=10):
    """Return the top k docs as (score, doc) pairs, best first."""
    k = max(1, min(k, len(docs)))
    scores = matrix @ query_vector
    # argpartition is faster than a full sort; then sort just the top k.
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [(float(scores[i]), docs[i]) for i in top]
