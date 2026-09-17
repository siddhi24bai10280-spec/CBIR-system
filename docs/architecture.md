# System Architecture

```mermaid
flowchart TB
    subgraph CLI["CLI Layer (main.py)"]
        BI[build-index command]
        Q[query command]
    end

    subgraph M1["Module 1: Preprocessing (preprocessing.py)"]
        LOAD[Load Image]
        RESIZE[Resize]
        DENOISE[Gaussian Denoise]
        ENHANCE[CLAHE Contrast Enhancement]
        LOAD --> RESIZE --> DENOISE --> ENHANCE
    end

    subgraph SEG["Segmentation (segmentation.py)"]
        GC[GrabCut Foreground Extraction]
    end

    subgraph M2["Module 2: Feature Extraction (feature_extraction.py)"]
        COLOR[Color Histogram - HSV]
        EDGE[Canny Edge Density]
        CORNER[Harris Corner Stats]
        ORIENT[Gradient Orientation Histogram]
        CONCAT[Concatenate to Feature Vector]
        COLOR --> CONCAT
        EDGE --> CONCAT
        CORNER --> CONCAT
        ORIENT --> CONCAT
    end

    subgraph IDX["Indexing (database.py + clustering.py)"]
        KMEANS[K-Means Clustering]
        STORE[(Feature Index .pkl)]
        KMEANS --> STORE
    end

    subgraph M3["Module 3: Similarity Search (similarity_search.py)"]
        DIST[Compute Distance - cosine/euclidean]
        RANK[Rank Top-N]
        DIST --> RANK
    end

    OUT[Result Grid Image - utils.py]

    BI --> M1
    ENHANCE -.optional.-> GC
    M1 --> M2
    M2 --> KMEANS
    Q --> M1
    M2 --> DIST
    STORE --> DIST
    RANK --> OUT
```

**Layers:**
- **CLI Layer** — user-facing entry point (`build-index`, `query` subcommands)
- **Module 1 (Preprocessing)** — Unit 1 concepts: convolution-based filtering, histogram-based enhancement
- **Segmentation** — optional GrabCut step (Unit 3: graph-cut segmentation)
- **Module 2 (Feature Extraction)** — Unit 3 concepts: edges, corners, orientation histograms
- **Indexing** — Unit 4 concept: K-Means clustering for faster retrieval
- **Module 3 (Similarity Search)** — distance computation and ranking
- **Output** — visual result grid for inspection
