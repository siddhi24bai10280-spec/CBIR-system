"""
preprocessing.py
-----------------
Module 1: Preprocessing & Enhancement
Maps to Unit 1 of the syllabus: Fundamentals of Image Formation,
Convolution & Filtering, Image Enhancement, Restoration, Histogram
Processing.

Responsibilities:
    - Load an image from disk safely (error handling / validation)
    - Resize to a canonical size so all feature vectors are comparable
    - Denoise using Gaussian filtering (convolution-based low-pass filter)
    - Enhance contrast using histogram equalization on the luminance
      channel (CLAHE), which is more robust than global equalization
"""

from __future__ import annotations
import os
import cv2
import numpy as np


class ImageLoadError(Exception):
    """Raised when an image cannot be read or is invalid."""


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR NumPy array.

    Raises:
        ImageLoadError: if the file does not exist or cannot be decoded.
    """
    if not os.path.isfile(path):
        raise ImageLoadError(f"File not found: {path}")

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(f"Could not decode image (unsupported/corrupt file): {path}")
    return image


def resize_image(image: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Resize an image to a canonical size using area interpolation
    (best for shrinking, avoids aliasing artifacts)."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def denoise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur (low-pass convolution filter) to suppress
    high-frequency sensor noise before feature extraction."""
    if kernel_size % 2 == 0:
        kernel_size += 1  # kernel size must be odd
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance local contrast using CLAHE (Contrast Limited Adaptive
    Histogram Equalization) applied to the L channel of LAB color
    space. This avoids over-amplifying noise the way global histogram
    equalization can."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)

    merged = cv2.merge((l_eq, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_pipeline(path: str, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Full Module 1 pipeline: load -> resize -> denoise -> enhance.

    Input:  path to an image file on disk
    Output: a cleaned, normalized BGR image ready for feature extraction
    """
    image = load_image(path)
    image = resize_image(image, size)
    image = denoise(image, kernel_size=5)
    image = enhance_contrast(image)
    return image
