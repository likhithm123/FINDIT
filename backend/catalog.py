"""
Sample Catalog Definition and ChromaDB Initialization
"""
import os
import chromadb
from backend.embedder import image_to_embedding

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db_store"))
COLLECTION_NAME = "multimodal_image_search"

SAMPLE_ITEMS = [
    {
        "id": "asset_001",
        "filename": "sunset_beach.jpg",
        "title": "Golden Sunset Over Ocean Horizon",
        "category": "nature",
        "tags": ["sunset", "beach", "ocean", "waves", "warm", "golden", "sand"],
        "description": "A tranquil tropical coastline illuminated by the warm golden and orange glow of the setting sun."
    },
    {
        "id": "asset_002",
        "filename": "mountain_snow.jpg",
        "title": "Snowy Alpine Mountain Peak",
        "category": "nature",
        "tags": ["mountain", "snow", "peak", "alpine", "cold", "winter", "ice", "white"],
        "description": "Jagged alpine peaks enveloped in pristine white snow against a crisp mountain sky."
    },
    {
        "id": "asset_003",
        "filename": "forest_mist.jpg",
        "title": "Misty Pine Forest in Morning Fog",
        "category": "nature",
        "tags": ["forest", "trees", "pine", "mist", "fog", "green", "emerald", "nature"],
        "description": "Layers of evergreen pine tree silhouettes shrouded in cool atmospheric morning mist."
    },
    {
        "id": "asset_004",
        "filename": "modern_skyscraper.jpg",
        "title": "Modern Architectural Glass Skyscraper",
        "category": "architecture",
        "tags": ["skyscraper", "building", "glass", "modern", "tower", "architecture", "blue"],
        "description": "Soaring steel and reflective blue glass skyscraper showcasing contemporary architectural design."
    },
    {
        "id": "asset_005",
        "filename": "ancient_temple.jpg",
        "title": "Classical Stone Pillars & Ancient Temple",
        "category": "architecture",
        "tags": ["temple", "ancient", "stone", "columns", "monument", "historic", "pediment"],
        "description": "Historical sandstone monument featuring classical architectural columns and ornate carved pediment."
    },
    {
        "id": "asset_006",
        "filename": "cozy_coffee.jpg",
        "title": "Cozy Espresso Cup with Latte Art",
        "category": "objects",
        "tags": ["coffee", "cup", "mug", "latte", "espresso", "beans", "wood", "warm"],
        "description": "A ceramic cup filled with rich espresso and delicate foam latte art resting on a rustic wooden table."
    },
    {
        "id": "asset_007",
        "filename": "neon_night_city.jpg",
        "title": "Cyberpunk Neon Night City Street",
        "category": "urban",
        "tags": ["neon", "night", "city", "cyberpunk", "street", "rain", "dark", "magenta", "cyan"],
        "description": "Electrifying urban nightlife illuminated by glowing magenta and cyan neon signs reflecting on wet asphalt."
    },
    {
        "id": "asset_008",
        "filename": "desert_dunes.jpg",
        "title": "Golden Sahara Sand Dunes",
        "category": "nature",
        "tags": ["desert", "dunes", "sand", "golden", "warm", "amber", "sun"],
        "description": "Vast sweeping sand dunes carved by desert winds, glowing in warm amber and golden sunlight."
    },
    {
        "id": "asset_009",
        "filename": "vintage_camera.jpg",
        "title": "Vintage Chrome 35mm Rangefinder Camera",
        "category": "objects",
        "tags": ["camera", "vintage", "lens", "retro", "photography", "technology", "chrome"],
        "description": "Classic mechanical rangefinder camera featuring brushed chrome top plate and multi-coated glass lens."
    },
    {
        "id": "asset_010",
        "filename": "coral_reef.jpg",
        "title": "Tropical Coral Reef and Marine Life",
        "category": "nature",
        "tags": ["coral", "reef", "ocean", "underwater", "tropical", "fish", "turquoise", "sea"],
        "description": "Vibrant underwater marine sanctuary featuring bright turquoise waters and colorful branching coral formations."
    },
    {
        "id": "asset_011",
        "filename": "autumn_park.jpg",
        "title": "Autumn Park Path with Golden Leaves",
        "category": "nature",
        "tags": ["autumn", "park", "leaves", "fall", "trees", "red", "orange", "yellow"],
        "description": "Curving stone pathway bordered by maple trees with brilliant scarlet, amber, and golden autumn foliage."
    },
    {
        "id": "asset_012",
        "filename": "minimal_interior.jpg",
        "title": "Minimalist Scandinavian Interior with Wooden Chair",
        "category": "architecture",
        "tags": ["interior", "minimal", "room", "chair", "wood", "scandinavian", "white", "clean"],
        "description": "Sunlit minimalist Scandinavian interior featuring clean white surfaces, natural oak flooring, and wooden chair."
    }
]

def get_chroma_client():
    os.makedirs(DB_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=DB_PATH)

def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def seed_database(force_reseed=False):
    collection = get_collection()
    count = collection.count()
    if count > 0 and not force_reseed:
        print(f"ChromaDB collection already initialized with {count} vectors.")
        return count

    if force_reseed:
        print("Clearing existing ChromaDB collection...")
        # Get all ids and delete
        existing = collection.get()
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    images_dir = os.path.join(base_dir, "static", "images")

    ids = []
    embeddings = []
    metadatas = []
    documents = []

    print(f"Indexing {len(SAMPLE_ITEMS)} sample assets into ChromaDB HNSW index...")
    for item in SAMPLE_ITEMS:
        image_path = os.path.join(images_dir, item["filename"])
        if not os.path.exists(image_path):
            print(f"Warning: Image file missing at {image_path}, skipping.")
            continue
        
        meta = {
            "title": item["title"],
            "category": item["category"],
            "tags": ", ".join(item["tags"]),
            "filename": item["filename"],
            "image_url": f"/static/images/{item['filename']}"
        }
        
        vec = image_to_embedding(image_path, meta)
        
        ids.append(item["id"])
        embeddings.append(vec)
        metadatas.append(meta)
        documents.append(item["description"])

    if ids:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"Successfully indexed {len(ids)} vectors into ChromaDB '{COLLECTION_NAME}' collection.")
    return len(ids)

if __name__ == "__main__":
    seed_database(force_reseed=True)
