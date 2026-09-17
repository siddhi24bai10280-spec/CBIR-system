# CBIR System — Content-Based Image Retrieval

A classical computer vision pipeline that finds visually similar images
from a dataset given a query image — no deep learning, just image
formation, filtering, feature extraction and similarity search.

Built as a course project applying the following syllabus units:

| Unit | Concepts used |
|---|---|
| Unit 1 — Digital Image Formation & Low-Level Processing | Gaussian filtering (convolution), CLAHE histogram-based contrast enhancement |
| Unit 3 — Feature Extraction & Segmentation | Canny edge detection, Harris corner detection, gradient orientation histograms (HOG-style), GrabCut (graph-cut) segmentation |
| Unit 4 — Pattern Analysis | K-Means clustering used to index the feature database for faster retrieval |

## Overview

The system has three functional modules that form a pipeline:

1. **Preprocessing & Enhancement** — loads, resizes, denoises and
   contrast-enhances every image so features are extracted from clean,
   comparable input.
2. **Feature Extraction** — computes a concatenated descriptor per
   image: HSV color histogram + Canny edge-density grid + Harris
   corner statistics + Sobel gradient orientation histogram.
3. **Similarity Search & Retrieval** — ranks the feature database
   against a query vector (cosine or Euclidean distance), optionally
   narrowing the search to the query's nearest K-Means cluster first.

```
Query image ─▶ [Preprocess] ─▶ [Extract Features] ─▶ [Compare vs Index] ─▶ Top-N results
                                                              ▲
Dataset folder ─▶ [Preprocess] ─▶ [Extract Features] ─▶ [Feature Index / K-Means clusters]
```

## Features

- End-to-end CLI: build an index from a folder of images, then query it
- Two similarity metrics: cosine and Euclidean distance
- K-Means–based cluster-narrowed search for better performance on larger datasets
- GrabCut-based foreground segmentation module (optional preprocessing step)
- Visual result grid output (query + ranked matches, saved as a single image)
- Graceful error handling for missing files, corrupt images and missing indexes
- 22 automated unit tests covering all core modules
- Synthetic sample-dataset generator so the project runs out of the box with no external downloads

## Technologies / Tools Used

- Python 3.12
- OpenCV (`opencv-python`) — image I/O, filtering, edge/corner detection, GrabCut
- NumPy — array/vector operations
- scikit-learn — K-Means clustering
- pytest — unit testing
- argparse — CLI interface

## Project Structure

```
cbir_system/
├── src/
│   ├── preprocessing.py       # Module 1: load, resize, denoise, enhance
│   ├── segmentation.py        # GrabCut foreground segmentation
│   ├── feature_extraction.py  # Module 2: color/edge/corner/orientation descriptors
│   ├── clustering.py          # K-Means indexing
│   ├── similarity_search.py   # Module 3: distance metrics + ranking
│   ├── database.py            # Build/save/load the feature index
│   ├── utils.py                # Logging, timing decorator, result-grid visualization
│   └── main.py                 # CLI entry point
├── scripts/
│   └── generate_sample_dataset.py   # Creates a synthetic demo dataset
├── tests/                       # 22 pytest unit tests
├── data/
│   ├── images/                  # Dataset to be indexed (generated or your own)
│   └── query/                   # Example query images
├── outputs/                     # Result grids saved here
├── docs/                        # Architecture / UML / workflow diagrams
├── requirements.txt
├── statement.md
└── README.md
```

## Steps to Install & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate a sample dataset (32 synthetic images across 4 shape categories)
python3 scripts/generate_sample_dataset.py

# 3. Build the feature index
python3 -m src.main build-index --dataset data/images --output data/index.pkl --clusters 4

# 4. Query with an example image
python3 -m src.main query --image data/query/query_circles.jpg --index data/index.pkl --top 5
```

To use your own dataset, drop images into `data/images/` (JPG/PNG/BMP
supported) and re-run the `build-index` step.

### CLI options

```
build-index --dataset <folder> --output <index.pkl> [--clusters K]
query        --image <path> --index <index.pkl> [--top N] [--metric cosine|euclidean]
             [--use-clusters] [--output <result_grid.jpg>]
```

## Instructions for Testing

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v
```

All 22 tests should pass, covering:
- Image loading and error handling (`test_preprocessing.py`)
- Feature vector shapes, normalization and self-distance (`test_feature_extraction.py`)
- Distance metrics and ranking correctness (`test_similarity_search.py`)
- Index build/save/load round-trips and missing-file handling (`test_database.py`)

## Screenshots

Example retrieval result (query on the left, ranked matches with distance scores):

`outputs/result_triangles.jpg` — shows a triangle query correctly retrieving
triangle images as its top matches ahead of other shapes.

## Non-Functional Requirements Addressed

- **Performance** — feature extraction and indexing are timed via a
  `@timed` decorator (logged in milliseconds); K-Means clustering
  narrows retrieval search space for larger datasets.
- **Reliability / Error Handling** — corrupt or missing files are
  caught with custom exceptions and skipped/logged rather than
  crashing the pipeline; CLI returns proper non-zero exit codes on failure.
- **Scalability** — cluster-narrowed search (`--use-clusters`) avoids
  a full linear scan as the dataset grows.
- **Maintainability** — each concern (preprocessing, features,
  clustering, search, storage) lives in its own module with a single
  responsibility and docstrings.
- **Usability** — a simple, documented CLI with sensible defaults and
  a saved visual result grid for quick inspection.
- **Logging/Monitoring** — structured logging throughout (`logging` module).
