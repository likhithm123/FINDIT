# Multimodal Vector Search Engine using ChromaDB
### DBMS Case Study: High-Dimensional Vector Storage & Sub-Linear ANN Retrieval

---

## 1. Executive Summary & Problem Formulation

In modern database management systems (DBMS), data has shifted dramatically from structured scalar records (integers, strings, timestamps) to **high-dimensional unstructured data** (natural language text, high-resolution imagery, audio, and video).

### The Limits of Traditional DBMS
1. **Relational B-Trees ($O(\log N)$)**: B-Tree indexes rely on strict, one-dimensional scalar sorting ($A < B < C$). They cannot index or order high-dimensional vectors ($\mathbb{R}^D$) because multidimensional space lacks a single natural total order.
2. **Inverted Indexes (TF-IDF / BM25)**: Keyword inverted indexes rely on exact lexicon matches. They fail to resolve semantic intent, synonyms, or cross-modal relationships (e.g., matching the text query *"serene ocean waves at sunset"* with an image containing an orange horizon and water without textual metadata).
3. **Brute-Force Vector Scans ($O(N \cdot D)$)**: Computing exact Euclidean ($L_2$) or Cosine distance across $N$ stored records in $D$ dimensions scales linearly. For collections of $10^6$ to $10^9$ assets, an $O(N \cdot D)$ exhaustive scan introduces prohibitive CPU and memory bandwidth bottlenecks.

### The Solution: Vector Databases (ChromaDB)
This project implements a **Multimodal Vector Search Engine** with **ChromaDB** at its architectural core. By mapping text phrases and image pixel distributions into a continuous, 128-dimensional unit hypersphere ($\mathbb{R}^{128}$), geometric spatial proximity directly reflects semantic similarity. ChromaDB couples **Approximate Nearest Neighbor (ANN)** graph indexing (**HNSW**) with **SQLite metadata persistence** to achieve sub-linear ($O(\log N)$) retrieval latency in **$< 2\text{ ms}$**.

---

## 2. ChromaDB Storage Architecture (Under the Hood)

ChromaDB uses a **hybrid dual-engine storage model**, separating relational metadata management from high-performance vector graph traversal.

```
                           +-------------------------------------------------+
                           |           CHROMADB STORAGE DIRECTORY            |
                           |              (./chroma_db_store)                |
                           +------------------------+------------------------+
                                                    |
                     +------------------------------+------------------------------+
                     |                                                             |
                     v                                                             v
+------------------------------------------+                 +------------------------------------------+
|      RELATIONAL METADATA ENGINE          |                 |          HNSW VECTOR GRAPH ENGINE        |
|          (chroma.sqlite3)                |                 |      (<uuid>/ Segment Directory)         |
+------------------------------------------+                 +------------------------------------------+
| • collections: Collection definitions    |                 | • data_level0.bin: Continuous raw float  |
| • embeddings: Document texts & IDs       |                 |   arrays (128-D dense vectors in R^128)  |
| • embedding_metadata: Key-value metadata |                 | • link_lists.bin: Multi-layer graph      |
| • segments: Vector index registry        |                 |   edge pointers (nearest neighbor graph) |
| • embedding_fulltext_search: SQLite FTS5 |                 | • header.bin: Distance metric (cosine),  |
|                                          |                 |   dimensionality (D=128), entry points   |
| Role: ACID transactions, record mapping, |                 | • length.bin: Element allocation counts  |
| and single-stage scalar WHERE filtering. |                 | Role: Sub-linear O(log N) ANN traversal  |
+------------------------------------------+                 +------------------------------------------+
```

### Physical Directory Layout
On disk, ChromaDB persists data directly in [`chroma_db_store/`](file:///Users/likhithmedarametla/Documents/AI/dbms%20case%20study%20pro/chroma_db_store):

```text
chroma_db_store/
├── chroma.sqlite3                           # SQLite relational store (schemas, docs, metadata)
└── 398c2912-29f9-4533-9630-86b1a4a8d18c/   # Vector Index Segment Folder (hnswlib)
    ├── data_level0.bin                      # 128-D floating-point vector coordinates
    ├── link_lists.bin                       # Graph links / adjacency lists across HNSW layers
    ├── header.bin                           # Index configuration, metric space, entry points
    └── length.bin                           # Allocation and capacity trackers
```

### SQLite Schema Breakdown (`chroma.sqlite3`)
| Table Name | Purpose | Key Attributes |
| :--- | :--- | :--- |
| `collections` | Defines vector namespaces / collections | `id`, `name`, `topic`, `dimension` |
| `embeddings` | Maps unique asset IDs to internal sequence IDs | `id`, `segment_id`, `created_at` |
| `embedding_metadata` | Stores scalar attributes for single-stage filtering | `id`, `key` (`category`, `tags`), `string_value` |
| `segments` | Binds collections to their underlying HNSW binary segment | `id`, `type` (`urn:chroma:segment/vector/hnsw-lib`) |
| `embedding_fulltext_search` | SQLite FTS5 virtual table for keyword indexing | Text tokens, document bodies |

---

## 3. End-to-End System Workflow

```
                                    [ User Search Input ]
                              (Text Query or Uploaded Image)
                                             |
                                             v
                         +---------------------------------------+
                         |      Multimodal Feature Encoder       |
                         |         (backend/embedder.py)         |
                         +-------------------+-------------------+
                                             |
                                             v
                         +---------------------------------------+
                         |     Dense Query Vector in R^128       |
                         |            ||q||_2 = 1.0              |
                         +-------------------+-------------------+
                                             |
                                             v
     +-------------------------------------------------------------------------------+
     |                        CHROMADB VECTOR ENGINE EXECUTION                       |
     |                                                                               |
     |   1. Query HNSW Multi-Layer Graph (Layer 2 -> Layer 1 -> Layer 0)             |
     |   2. Compute Cosine Distance: dist(q, d) = 1 - (q · d)                        |
     |   3. Apply Metadata Predicate: WHERE category = 'nature'                      |
     |   4. Select Top-K Ranked Neighbors via Priority Queue                         |
     +---------------------------------------+---------------------------------------+
                                             |
                                             v
                         +---------------------------------------+
                         |       Ranked Nearest Neighbors        |
                         |     (Score %, Distance d, Assets)     |
                         +---------------------------------------+
```

### 1. Data Ingestion & Feature Extraction
- **Image Embedding**: The encoder inspects the image asset:
  - 3D spatial color histograms (RGB & HSV distributions).
  - Spatial composition (e.g., upper sky luminance vs. lower terrain/sand/table).
  - Edge and texture density gradients (Laplacian variance).
  - Cross-modal tag and category projections.
- **Text Embedding**: Natural language queries are tokenized, matched against semantic concepts (e.g., *"sunset"*, *"mountain"*, *"neon"*, *"espresso"*), and mapped into correlated coordinates.
- **Normalization**: Every vector is projected onto the unit hypersphere:
  $$\vec{v} = \frac{\vec{v}}{\|\vec{v}\|_2} \implies \|\vec{v}\|_2 = 1.0$$

### 2. HNSW Graph Insertion
ChromaDB’s C++ core (`hnswlib`) inserts each vector into the Hierarchical Navigable Small World graph:
- It probabilistically assigns the vector an entry layer height.
- It calculates distance to existing nodes and connects bidirectional edges (`link_lists.bin`).

### 3. Query Execution & Graph Traversal
When a query arrives:
1. ChromaDB starts at the top-most sparse layer of the HNSW graph.
2. It greedily steps to the neighbor node closest to query vector $Q$.
3. Upon reaching a local minimum at the current layer, it steps down to the next denser layer.
4. At **Layer 0**, it performs a localized proximity search to collect the top-$k$ candidate nodes.
5. If a scalar filter (`where={"category": "nature"}`) is specified, candidates failing the predicate are filtered during traversal.
6. Returns sorted matches, exact distances, document text, and metadata.

---

## 4. Mathematical Foundations

### 1. Cosine Similarity vs. Cosine Distance
Because all embeddings are normalized to unit length ($\|A\|_2 = \|B\|_2 = 1$), the inner product equals the cosine of the angle $\theta$ between them:

$$\cos(\theta) = \frac{A \cdot B}{\|A\|_2 \|B\|_2} = A \cdot B$$

In ChromaDB (`hnsw:space = "cosine"`), distance is defined as:

$$\text{Distance}(A, B) = 1 - \cos(\theta) \in [0, 2]$$

Thus, the similarity percentage presented in the UI is:

$$\text{Similarity Score} = \left(1 - \text{Distance}\right) \times 100\%$$

- **Identical Vectors**: $\theta = 0^\circ \implies \text{Distance} = 0.0 \implies \text{Similarity} = 100\%$
- **Orthogonal Vectors**: $\theta = 90^\circ \implies \text{Distance} = 1.0 \implies \text{Similarity} = 0\%$
- **Opposite Vectors**: $\theta = 180^\circ \implies \text{Distance} = 2.0 \implies \text{Similarity} = 0\%$

### 2. Sub-Linear Scalability: ANN vs. Exact Search

```
   [ Layer 2 ]       (Node A) --------------------------> (Node B)        * Coarse Long-range Skips
                        |                                    |
                        v                                    v
   [ Layer 1 ]       (Node A) ---------> (Node C) ------> (Node B)        * Intermediate Routing
                        |                   |                |
                        v                   v                v
   [ Layer 0 ]       (Node A) -> (D) -> (Node C) -> (E) -> (Node B)       * Dense Full-Proximity Graph
```

| Strategy | Search Complexity | Insertion Complexity | Recall | Read Latency ($N=10^5$) |
| :--- | :--- | :--- | :--- | :--- |
| **Exact (Flat / Brute-Force)** | $O(N \cdot D)$ | $O(1)$ | $100\%$ | $\sim 80 - 150 \text{ ms}$ |
| **ChromaDB HNSW (ANN)** | **$O(\log N)$** | $O(M \log N)$ | $> 99\%$ | **$\sim 1 - 2 \text{ ms}$** |

---

## 5. How to Inspect & Visualize the Database

### Method A: Web UI Interactive 2D Vector Space Map
1. Launch the application: `./run.sh`
2. Open [http://127.0.0.1:8000](http://127.0.0.1:8000)
3. Click the **"2D Vector Space Map"** tab:
   - The backend runs **Principal Component Analysis (PCA)** (`backend/app.py` -> `/api/visualize`) to project the 128-D vectors down to a 2D Euclidean coordinate system.
   - You can visually observe semantic clustering: warm nature scenes cluster on the right, cold/alpine mountains on the top, and urban/architecture on the left.

### Method B: Web UI Vector Store Explorer & Inspector
- Click the **"Vector Store Explorer"** tab to view all records currently in ChromaDB.
- Click the **"Vector"** button on any card or table row to open the **Vector Inspector Modal**, revealing the exact coordinate values $\vec{v} \in \mathbb{R}^{128}$ and verification of unit norm ($\|\vec{v}\|_2 = 1.0$).

### Method C: Inspecting SQLite via Terminal CLI
Examine ChromaDB's underlying relational tables directly using `sqlite3`:

```bash
# View all tables created by ChromaDB
sqlite3 chroma_db_store/chroma.sqlite3 "SELECT name FROM sqlite_master WHERE type='table';"

# View total indexed records and segments
sqlite3 chroma_db_store/chroma.sqlite3 "SELECT count(*) FROM embeddings;"

# View scalar metadata associated with indexed assets
sqlite3 chroma_db_store/chroma.sqlite3 "SELECT id, key, string_value FROM embedding_metadata WHERE id='asset_001';"
```

### Method D: Programmatic Python Inspection
Run this script to fetch vectors directly from the persistent store:

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db_store")
collection = client.get_collection("multimodal_image_search")

print(f"Total Vectors in DB: {collection.count()}")

# Fetch records including raw embedding vectors and metadata
records = collection.get(include=["embeddings", "metadatas", "documents"])
for doc_id, meta, emb in zip(records["ids"][:3], records["metadatas"][:3], records["embeddings"][:3]):
    print(f"ID: {doc_id} | Category: {meta['category']} | Vector Sample: {emb[:5]}...")
```

---

## 6. Project Structure

```
.
├── backend/
│   ├── app.py              # FastAPI application & REST endpoints (/api/search, /api/visualize, etc.)
│   ├── embedder.py         # 128-D Multimodal continuous vector encoder for text & images
│   ├── catalog.py          # ChromaDB collection initialization, schema setup & database seeder
│   └── image_generator.py  # PIL script generating high-resolution curated asset imagery
├── chroma_db_store/        # ChromaDB on-disk persistent database (SQLite + HNSW segments)
├── static/
│   ├── index.html          # Minimalist light interface (Text/Image search, 2D Map, Explorer)
│   ├── style.css           # Clean white aesthetic design system
│   ├── app.js              # Frontend controller, canvas vector map renderer & telemetry
│   └── images/             # Visual asset image store
├── run.sh                  # One-click startup runner
└── README.md               # Technical database documentation
```

---

## 7. Quick Start Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Installed dependencies: `chromadb`, `fastapi`, `uvicorn`, `pillow`, `scikit-learn`, `numpy`, `python-multipart`

### Launching the Application
```bash
./run.sh
```

Or run via Uvicorn manually:
```bash
PYTHONPATH=. python3 -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
