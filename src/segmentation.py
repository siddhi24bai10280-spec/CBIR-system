"""
segmentation.py
-----------------
Supports Module 2. Maps to Unit 3 of the syllabus: Image Segmentation
(region-based / graph-cut approaches).

Uses GrabCut, an iterative graph-cut based segmentation algorithm, to
roughly separate the foreground object from the background before
feature extraction. This reduces background clutter influencing the
similarity search.
"""

from __future__ import annotations
import cv2
import numpy as np


def segment_foreground(image: np.ndarray, margin_ratio: float = 0.08) -> np.ndarray:
    """Segment the dominant foreground object using GrabCut.

    A rectangle covering the central region of the image (leaving a
    small margin) is used as the initial foreground hint, since we do
    not have manual annotations for an automated pipeline.

    Input:  BGR image
    Output: BGR image with background pixels set to black
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterCount=5,
                    mode=cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # GrabCut can fail on degenerate/very small images; fall back
        # to returning the original image unsegmented rather than crashing.
        return image

    binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
    segmented = image * binary_mask[:, :, np.newaxis]
    return segmented
