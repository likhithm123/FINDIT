"""
FastAPI Backend for ChromaDB Multimodal Vector Search Engine
"""
import os
import time
import shutil
import uuid
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.catalog import get_collection, seed_database, DB_PATH, COLLECTION_NAME, SAMPLE_ITEMS
from backend.embedder import text_to_embedding, image_to_embedding, EMBEDDING_DIM

app = FastAPI(title="Multimodal Vector Search Engine (ChromaDB)", version="1.0.0")

# Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(BASE_DIR, "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Ensure database is seeded on startup
@app.on_event("startup")
def startup_event():
    seed_database(force_reseed=False)

class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 6
    category: Optional[str] = "all"

class ImageIdSearchRequest(BaseModel):
    image_id: str
    top_k: int = 6
    category: Optional[str] = "all"

@app.get("/")
@app.head("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/stats")
def get_stats():
    """Returns database telemetry and HNSW configuration."""
    col = get_collection()
    count = col.count()
    return {
        "status": "connected",
        "engine": "ChromaDB (HNSW / SQLite)",
        "collection_name": COLLECTION_NAME,
        "total_vectors": count,
        "embedding_dimensions": EMBEDDING_DIM,
        "distance_metric": "Cosine Distance (hnsw:space = cosine)",
        "storage_mode": "Local Persistence",
        "storage_path": DB_PATH
    }

@app.get("/api/visualize")
def get_vector_space_2d():
    """Projects high-dimensional ChromaDB embeddings into 2D space via PCA for visual plotting."""
    from sklearn.decomposition import PCA
    col = get_collection()
    data = col.get(include=["embeddings", "metadatas", "documents"])
    if not data or not data.get("ids") or len(data["ids"]) == 0:
        return {"points": []}

    embeddings = data["embeddings"]
    if len(embeddings) >= 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
    else:
        coords = [[0.0, 0.0]]

    points = []
    for i in range(len(data["ids"])):
        meta = data["metadatas"][i] if data.get("metadatas") else {}
        points.append({
            "id": data["ids"][i],
            "x": round(float(coords[i][0]), 4),
            "y": round(float(coords[i][1]), 4),
            "title": meta.get("title", data["ids"][i]),
            "category": meta.get("category", "general"),
            "image_url": meta.get("image_url", "")
        })
    return {"points": points, "dimensions": 2, "original_dim": EMBEDDING_DIM}

@app.get("/api/catalog")
def get_catalog():
    """Fetches all indexed records from ChromaDB."""
    col = get_collection()
    data = col.get(include=["metadatas", "documents", "embeddings"])
    items = []
    if data and data.get("ids"):
        for i in range(len(data["ids"])):
            meta = data["metadatas"][i] if data.get("metadatas") else {}
            emb = data["embeddings"][i] if data.get("embeddings") is not None and len(data["embeddings"]) > i else None
            items.append({
                "id": data["ids"][i],
                "title": meta.get("title", data["ids"][i]),
                "category": meta.get("category", "general"),
                "tags": [t.strip() for t in meta.get("tags", "").split(",") if t.strip()],
                "image_url": meta.get("image_url", ""),
                "description": data["documents"][i] if data.get("documents") else "",
                "embedding_preview": [round(float(x), 4) for x in emb[:6]] if emb is not None else []
            })
    return {"items": items, "count": len(items)}

@app.post("/api/search/text")
def search_text(payload: TextSearchRequest):
    """Executes Text-to-Image Approximate Nearest Neighbor search via ChromaDB."""
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    col = get_collection()
    total_count = col.count()
    if total_count == 0:
        return {"results": [], "latency_ms": 0.0, "query_vector": []}

    top_k = min(max(payload.top_k, 1), total_count)
    
    t0 = time.perf_counter()
    query_vector = text_to_embedding(query)
    
    where_filter = None
    if payload.category and payload.category.lower() != "all":
        where_filter = {"category": payload.category.lower()}

    try:
        results = col.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "distances", "documents"]
        )
    except Exception as e:
        # Fallback without where if no items match category
        results = col.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["metadatas", "distances", "documents"]
        )
    
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    
    formatted_results = []
    if results and results.get("ids") and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            dist = float(results["distances"][0][i])
            # In ChromaDB cosine distance: distance = 1 - cos(theta)
            # Similarity is cos(theta) = 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - dist))
            meta = results["metadatas"][0][i]
            doc = results["documents"][0][i]
            
            formatted_results.append({
                "id": doc_id,
                "similarity": round(similarity, 4),
                "similarity_percent": round(similarity * 100, 1),
                "distance": round(dist, 4),
                "title": meta.get("title", doc_id),
                "category": meta.get("category", "general"),
                "tags": [t.strip() for t in meta.get("tags", "").split(",") if t.strip()],
                "image_url": meta.get("image_url", ""),
                "description": doc
            })

    return {
        "query": query,
        "search_type": "text-to-image",
        "top_k": top_k,
        "latency_ms": latency_ms,
        "query_vector_sample": [round(float(x), 4) for x in query_vector[:8]],
        "results": formatted_results
    }

@app.post("/api/search/image-id")
def search_by_image_id(payload: ImageIdSearchRequest):
    """Executes Image-to-Image search using an existing asset's vector in ChromaDB."""
    col = get_collection()
    asset = col.get(ids=[payload.image_id], include=["embeddings", "metadatas"])
    if not asset or not asset.get("ids") or len(asset["ids"]) == 0:
        raise HTTPException(status_code=404, detail="Image ID not found in database")

    query_vector = asset["embeddings"][0]
    total_count = col.count()
    top_k = min(max(payload.top_k, 1), total_count)
    
    t0 = time.perf_counter()
    where_filter = None
    if payload.category and payload.category.lower() != "all":
        where_filter = {"category": payload.category.lower()}

    try:
        results = col.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "distances", "documents"]
        )
    except Exception:
        results = col.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["metadatas", "distances", "documents"]
        )
    
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    
    formatted_results = []
    if results and results.get("ids") and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            dist = float(results["distances"][0][i])
            similarity = max(0.0, min(1.0, 1.0 - dist))
            meta = results["metadatas"][0][i]
            doc = results["documents"][0][i]
            
            formatted_results.append({
                "id": doc_id,
                "similarity": round(similarity, 4),
                "similarity_percent": round(similarity * 100, 1),
                "distance": round(dist, 4),
                "title": meta.get("title", doc_id),
                "category": meta.get("category", "general"),
                "tags": [t.strip() for t in meta.get("tags", "").split(",") if t.strip()],
                "image_url": meta.get("image_url", ""),
                "description": doc
            })

    return {
        "query_image_id": payload.image_id,
        "search_type": "image-to-image",
        "top_k": top_k,
        "latency_ms": latency_ms,
        "query_vector_sample": [round(float(x), 4) for x in query_vector[:8]],
        "results": formatted_results
    }

@app.post("/api/search/upload")
async def search_uploaded_image(
    file: UploadFile = File(...),
    top_k: int = Form(6),
    category: str = Form("all")
):
    """Executes Image-to-Image search from an uploaded image file."""
    temp_filename = f"query_{uuid.uuid4().hex[:8]}_{file.filename}"
    temp_path = os.path.join(IMAGES_DIR, temp_filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        t0 = time.perf_counter()
        query_vector = image_to_embedding(temp_path)
        col = get_collection()
        total_count = col.count()
        actual_top_k = min(max(top_k, 1), max(total_count, 1))

        where_filter = None
        if category and category.lower() != "all":
            where_filter = {"category": category.lower()}

        try:
            results = col.query(
                query_embeddings=[query_vector],
                n_results=actual_top_k,
                where=where_filter,
                include=["metadatas", "distances", "documents"]
            )
        except Exception:
            results = col.query(
                query_embeddings=[query_vector],
                n_results=actual_top_k,
                include=["metadatas", "distances", "documents"]
            )
        
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        
        formatted_results = []
        if results and results.get("ids") and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                dist = float(results["distances"][0][i])
                similarity = max(0.0, min(1.0, 1.0 - dist))
                meta = results["metadatas"][0][i]
                doc = results["documents"][0][i]
                
                formatted_results.append({
                    "id": doc_id,
                    "similarity": round(similarity, 4),
                    "similarity_percent": round(similarity * 100, 1),
                    "distance": round(dist, 4),
                    "title": meta.get("title", doc_id),
                    "category": meta.get("category", "general"),
                    "tags": [t.strip() for t in meta.get("tags", "").split(",") if t.strip()],
                    "image_url": meta.get("image_url", ""),
                    "description": doc
                })

        return {
            "query_image_url": f"/static/images/{temp_filename}",
            "search_type": "image-upload-to-image",
            "top_k": actual_top_k,
            "latency_ms": latency_ms,
            "query_vector_sample": [round(float(x), 4) for x in query_vector[:8]],
            "results": formatted_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/index")
async def index_new_asset(
    title: str = Form(...),
    category: str = Form("nature"),
    tags: str = Form(""),
    description: str = Form(""),
    file: UploadFile = File(...)
):
    """Indexes a newly uploaded asset into ChromaDB."""
    asset_id = f"custom_{uuid.uuid4().hex[:6]}"
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    filename = f"{asset_id}{ext}"
    image_path = os.path.join(IMAGES_DIR, filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    metadata = {
        "title": title.strip(),
        "category": category.strip().lower(),
        "tags": ", ".join(tag_list),
        "filename": filename,
        "image_url": f"/static/images/{filename}"
    }

    vector = image_to_embedding(image_path, metadata)
    doc_text = description.strip() or f"{title}. Category: {category}. Tags: {', '.join(tag_list)}"

    col = get_collection()
    col.upsert(
        ids=[asset_id],
        embeddings=[vector],
        metadatas=[metadata],
        documents=[doc_text]
    )

    return {
        "status": "success",
        "indexed_id": asset_id,
        "title": title,
        "category": category,
        "total_count": col.count(),
        "embedding_sample": [round(float(x), 4) for x in vector[:8]]
    }

@app.delete("/api/items/{item_id}")
def delete_item(item_id: str):
    """Deletes an item from ChromaDB."""
    col = get_collection()
    col.delete(ids=[item_id])
    return {"status": "deleted", "id": item_id, "remaining": col.count()}

@app.post("/api/reseed")
def reseed_catalog():
    """Resets the collection to default 12 curated assets."""
    count = seed_database(force_reseed=True)
    return {"status": "reseeded", "total_vectors": count}
