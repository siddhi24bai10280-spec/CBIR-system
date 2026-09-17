import numpy as np
import pytest

from src import feature_extraction


@pytest.fixture
def dummy_image():
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)


def test_color_histogram_shape_and_normalization(dummy_image):
    hist = feature_extraction.extract_color_histogram(dummy_image, bins=(4, 4, 4))
    assert hist.shape == (64,)
    assert np.isclose(np.linalg.norm(hist), 1.0, atol=1e-3) or np.linalg.norm(hist) == 0


def test_edge_features_are_normalized(dummy_image):
    vec = feature_extraction.extract_edge_features(dummy_image, bins=16)
    norm = np.linalg.norm(vec)
    assert norm == pytest.approx(1.0, abs=1e-3) or norm == 0


def test_corner_features_length(dummy_image):
    vec = feature_extraction.extract_corner_features(dummy_image)
    assert vec.shape == (4,)


def test_orientation_histogram_length(dummy_image):
    vec = feature_extraction.extract_orientation_histogram(dummy_image, n_bins=9)
    assert vec.shape == (9,)


def test_combined_feature_vector_is_concatenation(dummy_image):
    combined = feature_extraction.extract_feature_vector(dummy_image)
    color = feature_extraction.extract_color_histogram(dummy_image)
    edges = feature_extraction.extract_edge_features(dummy_image)
    corners = feature_extraction.extract_corner_features(dummy_image)
    orientation = feature_extraction.extract_orientation_histogram(dummy_image)

    expected_len = len(color) + len(edges) + len(corners) + len(orientation)
    assert combined.shape[0] == expected_len
    assert combined.dtype == np.float32


def test_identical_images_have_zero_or_near_zero_self_distance(dummy_image):
    from src import similarity_search
    v1 = feature_extraction.extract_feature_vector(dummy_image)
    v2 = feature_extraction.extract_feature_vector(dummy_image)
    distance = similarity_search.compute_distance(v1, v2, metric="cosine")
    assert distance == pytest.approx(0.0, abs=1e-4)
