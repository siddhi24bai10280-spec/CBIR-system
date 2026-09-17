"""
clustering.py
---------------
Supports Module 3 (indexing/scalability). Maps to Unit 4 of the
syllabus: Pattern Analysis - Clustering (K-Means).

Rather than comparing a query against every image in the database
(linear scan), images are first grouped into K clusters using K-Means.
At query time, the query is compared to cluster centroids first, and
the similarity search is narrowed to the nearest cluster's members.
This is a simple but real illustration of using unsupervised learning
to speed up retrieval as a dataset scales (non-functional requirement:
performance / scalability).
"""

from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans


def build_clusters(feature_matrix: np.ndarray, k: int = 4, random_state: int = 42):
    """Cluster the feature database with K-Means.

    Input:  feature_matrix (n_images x n_features)
    Output: (labels, centroids) - cluster assignment per image and the
            K cluster centers
    """
    k = max(1, min(k, len(feature_matrix)))  # k cannot exceed n_samples
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(feature_matrix)
    return labels, model.cluster_centers_


def nearest_cluster(query_vector: np.ndarray, centroids: np.ndarray) -> int:
    """Return the index of the cluster whose centroid is closest to
    the query feature vector (Euclidean distance)."""
    distances = np.linalg.norm(centroids - query_vector, axis=1)
    return int(np.argmin(distances))
