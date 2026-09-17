"""
main.py
---------
Command-line entry point for the CBIR system. Wires together
Module 1 (preprocessing), Module 2 (feature extraction) and
Module 3 (similarity search) into two user-facing workflows:

    build-index   Preprocess + extract features for a folder of images
    query          Find the top-N images most similar to a query image

Usage:
    python -m src.main build-index --dataset data/images --output data/index.pkl
    python -m src.main query --image data/query/sample.jpg --index data/index.pkl --top 5
"""

from __future__ import annotations
import argparse
import logging
import sys

from src import preprocessing
from src import feature_extraction
from src import similarity_search
from src import database
from src import clustering
from src import utils

logger = logging.getLogger("cbir.main")


@utils.timed
def cmd_build_index(args: argparse.Namespace) -> None:
    database.build_database(args.dataset, args.output, n_clusters=args.clusters)


@utils.timed
def cmd_query(args: argparse.Namespace) -> None:
    index = database.load_database(args.index)

    query_image = preprocessing.preprocess_pipeline(args.image)
    query_vector = feature_extraction.extract_feature_vector(query_image)

    if args.use_clusters and len(index["cluster_centroids"]) > 1:
        cluster_id = clustering.nearest_cluster(query_vector, index["cluster_centroids"])
        mask = index["cluster_labels"] == cluster_id
        search_vectors = index["vectors"][mask]
        search_paths = [p for p, keep in zip(index["paths"], mask) if keep]
        logger.info("Searching within cluster %d (%d candidate images)",
                    cluster_id, len(search_paths))
    else:
        search_vectors = index["vectors"]
        search_paths = index["paths"]

    results = similarity_search.rank_similar(
        query_vector, search_vectors, search_paths,
        top_n=args.top, metric=args.metric,
    )

    if not results:
        print("No results found. Is the index empty?")
        return

    print(f"\nTop {len(results)} matches for '{args.image}' (metric={args.metric}):")
    for rank, (path, distance) in enumerate(results, start=1):
        print(f"  {rank}. {path}  (distance={distance:.4f})")

    output_path = utils.save_result_grid(args.image, results, args.output)
    print(f"\nResult grid saved to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cbir",
        description="Content-Based Image Retrieval using classical computer vision.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_p = subparsers.add_parser("build-index", help="Build a feature index from a folder of images.")
    build_p.add_argument("--dataset", required=True, help="Folder containing images to index.")
    build_p.add_argument("--output", required=True, help="Path to write the index (.pkl).")
    build_p.add_argument("--clusters", type=int, default=4, help="Number of K-Means clusters.")
    build_p.set_defaults(func=cmd_build_index)

    query_p = subparsers.add_parser("query", help="Find images similar to a query image.")
    query_p.add_argument("--image", required=True, help="Path to the query image.")
    query_p.add_argument("--index", required=True, help="Path to a previously built index (.pkl).")
    query_p.add_argument("--top", type=int, default=5, help="Number of results to return.")
    query_p.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    query_p.add_argument("--use-clusters", action="store_true",
                          help="Narrow the search to the nearest K-Means cluster first.")
    query_p.add_argument("--output", default="outputs/result_grid.jpg",
                          help="Where to save the visual result grid.")
    query_p.set_defaults(func=cmd_query)

    return parser


def main(argv: list[str] | None = None) -> int:
    utils.setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)
    except (preprocessing.ImageLoadError, FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
        logger.error(str(exc))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
