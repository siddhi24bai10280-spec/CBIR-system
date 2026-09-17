"""
feature_extraction.py
-----------------------
Module 2: Feature Extraction
Maps to Unit 3 of the syllabus: Edges (Canny), Corners (Harris),
Orientation Histograms (HOG-style), and general Feature Extraction.

Four complementary descriptors are computed and concatenated into a
single feature vector per image:
    1. Color histogram (HSV space)      -> captures color distribution
    2. Canny edge density histogram     -> captures shape/edge content
    3. Harris corner statistics         -> captures structural keypoints
    4. Gradient orientation histogram   -> lightweight HOG-style texture
"""

from __future__ import annotations
import cv2
import numpy as np


def extract_color_histogram(image: np.ndarray, bins: tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
    """Compute a normalized 3D color histogram in HSV space, flattened
    to a 1D vector. HSV is used because it separates chromaticity from
    illumination, making the descriptor more lighting-robust than RGB."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins),
                         [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_edge_features(image: np.ndarray, bins: int = 16) -> np.ndarray:
    """Detect edges with the Canny detector and summarize their spatial
    distribution as a coarse grid-based density histogram."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)

    grid = int(np.sqrt(bins))
    h, w = edges.shape
    cell_h, cell_w = h // grid, w // grid

    densities = []
    for i in range(grid):
        for j in range(grid):
            cell = edges[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            densities.append(np.mean(cell > 0) if cell.size else 0.0)

    vec = np.array(densities, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def extract_corner_features(image: np.ndarray) -> np.ndarray:
    """Detect corners with the Harris detector and summarize them as
    [corner_count_normalized, mean_response, std_response, max_response]."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_f = np.float32(gray)
    response = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
    response = cv2.dilate(response, None)

    threshold = 0.01 * response.max() if response.max() > 0 else 0
    corner_mask = response > threshold
    corner_count = np.sum(corner_mask)
    total_pixels = response.size

    strengths = response[corner_mask] if corner_count > 0 else np.array([0.0])
    vec = np.array([
        corner_count / total_pixels,
        float(np.mean(strengths)),
        float(np.std(strengths)),
        float(np.max(strengths)),
    ], dtype=np.float32)

    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def extract_orientation_histogram(image: np.ndarray, n_bins: int = 9) -> np.ndarray:
    """Lightweight HOG-style descriptor: compute gradient magnitude and
    orientation (Sobel derivatives) over the whole image and build a
    single orientation histogram weighted by gradient magnitude."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = angle % 180  # unsigned orientation

    hist, _ = np.histogram(angle, bins=n_bins, range=(0, 180), weights=magnitude)
    norm = np.linalg.norm(hist)
    return (hist / norm if norm > 0 else hist).astype(np.float32)


def extract_feature_vector(image: np.ndarray) -> np.ndarray:
    """Combine all descriptors into a single concatenated feature
    vector representing the image.

    Input:  preprocessed BGR image
    Output: 1D float32 NumPy array (the image's feature signature)
    """
    color = extract_color_histogram(image)
    edges = extract_edge_features(image)
    corners = extract_corner_features(image)
    orientation = extract_orientation_histogram(image)

    return np.concatenate([color, edges, corners, orientation]).astype(np.float32)
