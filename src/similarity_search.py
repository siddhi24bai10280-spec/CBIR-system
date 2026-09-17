"""
similarity_search.py
----------------------
Module 3: Similarity Search & Retrieval

Given a query feature vector and a database of feature vectors, rank
database images by similarity and return the top-N matches. Supports
two distance metrics and optional cluster-narrowed search (see
clustering.py) for better performance on larger datasets.
"""

from __future__ import annotations
import numpy as np


def compute_distance(vec1: np.ndarray, vec2: np.ndarray, metric: str = "cosine") -> float:
    """Compute the distance between two feature vectors.
    Lower is more similar for both metrics.
    """
    if metric == "euclidean":
        return float(np.linalg.norm(vec1 - vec2))

    if metric == "cosine":
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        cosine_similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(1.0 - cosine_similarity)  # convert similarity -> distance

    raise ValueError(f"Unsupported metric: {metric}. Use 'cosine' or 'euclidean'.")


def rank_similar(query_vector: np.ndarray,
                  database_vectors: np.ndarray,
                  database_paths: list[str],
                  top_n: int = 5,
                  metric: str = "cosine") -> list[tuple[str, float]]:
    """Rank all database images by similarity to the query vector.

    Input:
        query_vector:      feature vector of the query image
        database_vectors:  (n_images x n_features) array
        database_paths:    file paths corresponding to each row
        top_n:              how many results to return
        metric:             'cosine' or 'euclidean'
    Output:
        list of (image_path, distance) tuples, sorted best-first
    """
    if len(database_vectors) == 0:
        return []

    distances = [compute_distance(query_vector, db_vec, metric) for db_vec in database_vectors]
    ranked_indices = np.argsort(distances)[:top_n]

    return [(database_paths[i], distances[i]) for i in ranked_indices]
