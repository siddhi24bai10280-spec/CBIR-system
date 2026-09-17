import numpy as np
import cv2
import pytest

from src import preprocessing


@pytest.fixture
def sample_image_path(tmp_path):
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    path = tmp_path / "sample.jpg"
    cv2.imwrite(str(path), image)
    return str(path)


def test_load_image_success(sample_image_path):
    image = preprocessing.load_image(sample_image_path)
    assert image is not None
    assert image.shape[2] == 3


def test_load_image_missing_file_raises():
    with pytest.raises(preprocessing.ImageLoadError):
        preprocessing.load_image("does_not_exist.jpg")


def test_resize_image_changes_dimensions(sample_image_path):
    image = preprocessing.load_image(sample_image_path)
    resized = preprocessing.resize_image(image, size=(64, 64))
    assert resized.shape[:2] == (64, 64)


def test_denoise_preserves_shape(sample_image_path):
    image = preprocessing.load_image(sample_image_path)
    denoised = preprocessing.denoise(image)
    assert denoised.shape == image.shape


def test_enhance_contrast_preserves_shape(sample_image_path):
    image = preprocessing.load_image(sample_image_path)
    enhanced = preprocessing.enhance_contrast(image)
    assert enhanced.shape == image.shape


def test_preprocess_pipeline_end_to_end(sample_image_path):
    result = preprocessing.preprocess_pipeline(sample_image_path, size=(128, 128))
    assert result.shape == (128, 128, 3)
    assert result.dtype == np.uint8
