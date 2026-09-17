import numpy as np
import pytest

from src import similarity_search


def test_compute_distance_cosine_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert similarity_search.compute_distance(v, v, metric="cosine") == pytest.approx(0.0, abs=1e-6)


def test_compute_distance_euclidean_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert similarity_search.compute_distance(v, v, metric="euclidean") == pytest.approx(0.0, abs=1e-6)


def test_compute_distance_orthogonal_vectors_cosine():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    assert similarity_search.compute_distance(v1, v2, metric="cosine") == pytest.approx(1.0, abs=1e-6)


def test_compute_distance_invalid_metric_raises():
    v = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        similarity_search.compute_distance(v, v, metric="not_a_metric")


def test_rank_similar_returns_best_match_first():
    query = np.array([1.0, 0.0, 0.0])
    database = np.array([
        [0.0, 1.0, 0.0],   # far
        [0.9, 0.1, 0.0],   # close
        [-1.0, 0.0, 0.0],  # opposite
    ])
    paths = ["far.jpg", "close.jpg", "opposite.jpg"]

    results = similarity_search.rank_similar(query, database, paths, top_n=2, metric="cosine")

    assert len(results) == 2
    assert results[0][0] == "close.jpg"


def test_rank_similar_empty_database_returns_empty_list():
    query = np.array([1.0, 0.0])
    results = similarity_search.rank_similar(query, np.array([]), [], top_n=5)
    assert results == []
