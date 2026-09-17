import os
import numpy as np
import cv2
import pytest

from src import database


@pytest.fixture
def sample_dataset_dir(tmp_path):
    dataset_dir = tmp_path / "images"
    dataset_dir.mkdir()
    rng = np.random.default_rng(1)
    for i in range(6):
        image = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_dir / f"img_{i}.jpg"), image)
    return str(dataset_dir)


def test_list_images_finds_all_files(sample_dataset_dir):
    paths = database.list_images(sample_dataset_dir)
    assert len(paths) == 6


def test_list_images_missing_dir_raises():
    with pytest.raises(NotADirectoryError):
        database.list_images("no_such_directory")


def test_build_and_load_database_round_trip(sample_dataset_dir, tmp_path):
    output_path = str(tmp_path / "index.pkl")
    index = database.build_database(sample_dataset_dir, output_path, n_clusters=2)

    assert len(index["paths"]) == 6
    assert index["vectors"].shape[0] == 6
    assert os.path.isfile(output_path)

    loaded = database.load_database(output_path)
    assert loaded["paths"] == index["paths"]
    np.testing.assert_array_equal(loaded["vectors"], index["vectors"])


def test_load_database_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        database.load_database(str(tmp_path / "missing.pkl"))
