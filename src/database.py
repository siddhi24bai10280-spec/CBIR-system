"""
database.py
-------------
Handles building, saving and loading the feature index that backs the
retrieval system. Stores per-image feature vectors, file paths and
(optionally) cluster assignments in a single pickle file so the
expensive preprocessing + feature extraction step only has to run once
per dataset (performance requirement).
"""

from __future__ import annotations
import os
import pickle
import logging
import numpy as np

from src import preprocessing
from src import feature_extraction
from src import clustering

logger = logging.getLogger("cbir.database")

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(directory: str) -> list[str]:
    """Return sorted absolute paths of all supported images in a directory."""
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Dataset directory not found: {directory}")

    paths = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    if not paths:
        raise FileNotFoundError(f"No supported images found in: {directory}")
    return paths


def build_database(image_dir: str, output_path: str, n_clusters: int = 4) -> dict:
    """Run the full pipeline (preprocess -> extract features -> cluster)
    over every image in image_dir and persist the resulting index.

    Input:  path to a folder of images
    Output: dict index, also written to disk at output_path
    """
    paths = list_images(image_dir)
    vectors = []
    valid_paths = []

    for path in paths:
        try:
            image = preprocessing.preprocess_pipeline(path)
            vector = feature_extraction.extract_feature_vector(image)
            vectors.append(vector)
            valid_paths.append(path)
        except preprocessing.ImageLoadError as exc:
            # Skip unreadable files rather than crashing the whole build
            # (error handling / reliability requirement).
            logger.warning("Skipping unreadable image %s: %s", path, exc)

    if not vectors:
        raise RuntimeError("No valid images could be processed for the database.")

    feature_matrix = np.vstack(vectors)
    labels, centroids = clustering.build_clusters(feature_matrix, k=n_clusters)

    index = {
        "paths": valid_paths,
        "vectors": feature_matrix,
        "cluster_labels": labels,
        "cluster_centroids": centroids,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(index, f)

    logger.info("Database built: %d images indexed into %d clusters -> %s",
                len(valid_paths), len(centroids), output_path)
    return index


def load_database(path: str) -> dict:
    """Load a previously built feature index from disk."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Index file not found: {path}. Run 'build-index' first.")
    with open(path, "rb") as f:
        return pickle.load(f)
