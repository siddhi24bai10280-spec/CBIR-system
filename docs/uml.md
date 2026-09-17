# UML Diagrams

## Use Case Diagram

```mermaid
flowchart LR
    User((User))
    UC1([Build feature index from a dataset])
    UC2([Query with an image to find similar images])
    UC3([Choose similarity metric])
    UC4([Enable cluster-narrowed search])
    UC5([View saved result grid])

    User --> UC1
    User --> UC2
    UC2 --> UC3
    UC2 --> UC4
    UC2 --> UC5
```

## Class / Component Diagram

```mermaid
classDiagram
    class Preprocessing {
        +load_image(path) Image
        +resize_image(image, size) Image
        +denoise(image, kernel_size) Image
        +enhance_contrast(image) Image
        +preprocess_pipeline(path, size) Image
    }

    class Segmentation {
        +segment_foreground(image, margin_ratio) Image
    }

    class FeatureExtraction {
        +extract_color_histogram(image, bins) Vector
        +extract_edge_features(image, bins) Vector
        +extract_corner_features(image) Vector
        +extract_orientation_histogram(image, n_bins) Vector
        +extract_feature_vector(image) Vector
    }

    class Clustering {
        +build_clusters(feature_matrix, k) Labels, Centroids
        +nearest_cluster(query_vector, centroids) int
    }

    class SimilaritySearch {
        +compute_distance(vec1, vec2, metric) float
        +rank_similar(query_vector, db_vectors, paths, top_n, metric) List
    }

    class Database {
        +list_images(directory) List~str~
        +build_database(image_dir, output_path, n_clusters) dict
        +load_database(path) dict
    }

    class CBIR_CLI {
        +cmd_build_index(args)
        +cmd_query(args)
        +main(argv) int
    }

    CBIR_CLI --> Preprocessing
    CBIR_CLI --> FeatureExtraction
    CBIR_CLI --> Database
    CBIR_CLI --> SimilaritySearch
    CBIR_CLI --> Clustering
    Database --> Preprocessing
    Database --> FeatureExtraction
    Database --> Clustering
    FeatureExtraction ..> Segmentation : optional pre-step
```

## Sequence Diagram — Query Flow

```mermaid
sequenceDiagram
    actor User
    participant CLI as main.py
    participant Pre as Preprocessing
    participant FE as FeatureExtraction
    participant DB as Database
    participant CL as Clustering
    participant SS as SimilaritySearch
    participant U as Utils

    User->>CLI: query --image X --index idx.pkl
    CLI->>DB: load_database(idx.pkl)
    DB-->>CLI: index (vectors, paths, centroids)
    CLI->>Pre: preprocess_pipeline(X)
    Pre-->>CLI: cleaned image
    CLI->>FE: extract_feature_vector(image)
    FE-->>CLI: query_vector
    opt use-clusters enabled
        CLI->>CL: nearest_cluster(query_vector, centroids)
        CL-->>CLI: cluster_id
    end
    CLI->>SS: rank_similar(query_vector, vectors, paths, top_n)
    SS-->>CLI: ranked results
    CLI->>U: save_result_grid(query, results)
    U-->>CLI: output_path
    CLI-->>User: printed ranked list + saved image
```

## Note on ER Diagram

This project does not use a relational database — the feature index is
persisted as a single serialized Python dictionary (`.pkl` file)
containing image paths, feature vectors, and cluster assignments. No
ER diagram applies; see `database.py` for the index schema instead:

```
index = {
    "paths":             List[str]            # image file paths
    "vectors":           np.ndarray (N x D)    # feature vectors
    "cluster_labels":    np.ndarray (N,)       # K-Means cluster id per image
    "cluster_centroids": np.ndarray (K x D)    # cluster centers
}
```
