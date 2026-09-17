"""
utils.py
----------
Shared helpers: logging configuration, a performance-timing decorator,
and a routine to save a visual grid of retrieval results.
"""

from __future__ import annotations
import time
import logging
import functools
import os
import cv2
import numpy as np


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a consistent logging format for the whole project."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def timed(func):
    """Decorator that logs how long a function took to run.
    Used to expose the performance non-functional requirement."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("%s completed in %.2f ms", func.__name__, elapsed_ms)
        return result
    return wrapper


def save_result_grid(query_path: str, results: list[tuple[str, float]],
                      output_path: str, thumb_size: tuple[int, int] = (150, 150)) -> str:
    """Compose the query image and its top-N matches into a single
    labeled image grid and save it to disk for easy visual inspection.
    """
    def load_thumb(path: str, label: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((*thumb_size, 3), dtype=np.uint8)
        img = cv2.resize(img, thumb_size)
        canvas = np.full((thumb_size[1] + 25, thumb_size[0], 3), 255, dtype=np.uint8)
        canvas[:thumb_size[1], :, :] = img
        cv2.putText(canvas, label, (5, thumb_size[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        return canvas

    tiles = [load_thumb(query_path, "QUERY")]
    for path, distance in results:
        tiles.append(load_thumb(path, f"{os.path.basename(path)[:12]} ({distance:.2f})"))

    grid = np.hstack(tiles)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, grid)
    return output_path
