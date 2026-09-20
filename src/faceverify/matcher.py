import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors (dot product)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b))


def normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector; a zero vector is returned unchanged."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def mean_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """Average a list of embeddings and L2-normalize the result."""
    if not embeddings:
        raise ValueError("embeddings must not be empty")
    mean = np.mean(np.stack(embeddings), axis=0)
    return normalize(mean)


def verify(
    test_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    threshold: float,
) -> tuple[bool, float]:
    """Return (is_match, similarity) comparing two normalized embeddings."""
    sim = cosine_similarity(test_embedding, reference_embedding)
    return bool(sim >= threshold), sim
