import numpy as np
import pytest

from faceverify.matcher import (
    cosine_similarity,
    mean_embedding,
    normalize,
    verify,
)


def test_cosine_similarity_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_normalize_unit_norm():
    n = normalize(np.array([3.0, 4.0]))
    assert np.linalg.norm(n) == pytest.approx(1.0)


def test_normalize_zero_vector_unchanged():
    v = np.zeros(3)
    assert np.allclose(normalize(v), v)


def test_mean_embedding_normalized():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    m = mean_embedding([a, b])
    assert m.shape == (3,)
    assert np.linalg.norm(m) == pytest.approx(1.0)
    assert m[0] == pytest.approx(m[1])
    assert m[2] == pytest.approx(0.0)


def test_mean_embedding_empty_raises():
    with pytest.raises(ValueError):
        mean_embedding([])


def test_verify_match():
    ref = np.array([1.0, 0.0, 0.0])
    test = normalize(np.array([0.99, 0.14, 0.0]))
    is_match, sim = verify(test, ref, threshold=0.9)
    assert is_match is True
    assert sim == pytest.approx(cosine_similarity(test, ref))


def test_verify_no_match():
    ref = np.array([1.0, 0.0, 0.0])
    test = np.array([0.0, 1.0, 0.0])
    is_match, sim = verify(test, ref, threshold=0.9)
    assert is_match is False
    assert sim == pytest.approx(0.0)
