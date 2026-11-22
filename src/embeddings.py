"""Turn abstracts into TF-IDF vectors."""

import json
import math
import re
from pathlib import Path

import numpy as np

# Words in almost every abstract, so they carry no meaning here.
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for",
    "with", "by", "we", "this", "that", "it", "is", "are", "as",
    "from", "at", "be", "our", "their", "its", "which", "can",
    "also", "using", "used", "use", "based", "via", "into",
}

TOKEN = re.compile(r"[a-z][a-z0-9]+")


def tokenize(text):
    """Lowercase words, drop stopwords. Returns a list of tokens."""
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


def load_docs(path):
    """Load one JSON object per line: id, title, summary, published."""
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def build_vocab(docs, max_terms=5000, min_docs=2):
    """Pick the vocabulary: words in at least min_docs papers."""
    doc_counts = {}
    for doc in docs:
        text = doc.get("title", "") + " " + doc.get("summary", "")
        for word in set(tokenize(text)):
            doc_counts[word] = doc_counts.get(word, 0) + 1

    words = [w for w, c in doc_counts.items() if c >= min_docs]
    # Keep the most common words, so the matrix stays small.
    words.sort(key=lambda w: doc_counts[w], reverse=True)
    words = words[:max_terms]

    vocab = {word: i for i, word in enumerate(words)}
    n_docs = len(docs)
    idf = np.array(
        [math.log((1 + n_docs) / (1 + doc_counts[w])) + 1.0 for w in words],
        dtype=np.float32,
    )
    return vocab, idf


def vectorize(texts, vocab, idf):
    """Turn texts into a normalised TF-IDF matrix, one row per text."""
    matrix = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row, text in enumerate(texts):
        counts = {}
        for word in tokenize(text):
            col = vocab.get(word)
            if col is not None:
                counts[col] = counts.get(col, 0) + 1
        for col, count in counts.items():
            # log(1 + count): saying a word twice matters, but not twice as much.
            matrix[row, col] = math.log1p(count) * idf[col]
    # Normalise each row to length 1, so dot product = cosine similarity.
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return matrix / lengths


def build_index(data_path, out_dir):
    """Build the Step 1 index from a .jsonl corpus and save it to disk."""
    docs = load_docs(data_path)
    texts = [d.get("title", "") + " " + d.get("summary", "") for d in docs]
    vocab, idf = build_vocab(docs)
    matrix = vectorize(texts, vocab, idf)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", matrix)
    np.save(out / "idf.npy", idf)
    (out / "vocab.json").write_text(json.dumps(vocab))
    return docs, vocab, idf, matrix


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    docs, vocab, idf, matrix = build_index(
        here / "data" / "arxiv_1000.jsonl", here / "data"
    )
    print(f"docs: {len(docs)}")
    print(f"vocab size: {len(vocab)}")
    print(f"matrix shape: {matrix.shape}")
