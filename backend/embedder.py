"""
Multimodal Vector Embedder:
Generates dense 128-dimensional normalized unit embeddings in R^128 for text and images,
enabling joint Text-to-Image and Image-to-Image Approximate Nearest Neighbor (ANN) search.
"""
import os
import re
import math
import hashlib
import numpy as np
from PIL import Image

EMBEDDING_DIM = 128

# Semantic concept dictionary mapping keywords to feature subspaces
CONCEPT_SPACE = {
    # Nature & Landscapes
    "sunset": [0, 1, 2, 10, 11],
    "beach": [1, 2, 3, 12, 13],
    "ocean": [2, 3, 4, 13, 14],
    "sea": [2, 3, 4, 13, 14],
    "waves": [3, 4, 14, 15],
    "mountain": [5, 6, 7, 16, 17],
    "snow": [6, 7, 8, 17, 18],
    "alpine": [6, 7, 8, 17, 18],
    "cold": [7, 8, 18, 19],
    "winter": [7, 8, 18, 19],
    "ice": [7, 8, 18, 19],
    "forest": [8, 9, 20, 21],
    "trees": [8, 9, 20, 21],
    "pine": [9, 20, 21, 22],
    "mist": [9, 10, 22, 23],
    "fog": [9, 10, 22, 23],
    "desert": [10, 11, 24, 25],
    "dunes": [10, 11, 24, 25],
    "sand": [1, 10, 24, 25],
    "coral": [12, 13, 26, 27],
    "reef": [12, 13, 26, 27],
    "underwater": [13, 14, 26, 27],
    "fish": [13, 26, 27],
    "autumn": [14, 15, 28, 29],
    "fall": [14, 15, 28, 29],
    "park": [14, 15, 28, 29],
    "leaves": [15, 28, 29],

    # Architecture & Interior
    "skyscraper": [30, 31, 32, 40, 41],
    "building": [30, 31, 32, 40, 41],
    "glass": [31, 32, 41, 42],
    "modern": [30, 33, 42, 43],
    "tower": [30, 32, 41, 44],
    "temple": [34, 35, 45, 46],
    "ancient": [34, 35, 45, 46],
    "stone": [34, 36, 46, 47],
    "columns": [35, 36, 46, 47],
    "monument": [35, 37, 47, 48],
    "interior": [38, 39, 49, 50],
    "room": [38, 39, 49, 50],
    "minimal": [38, 39, 50, 51],
    "scandinavian": [38, 39, 51, 52],
    "chair": [39, 49, 52, 53],
    "wood": [39, 49, 53, 54],

    # Urban & Objects
    "neon": [55, 56, 57, 65, 66],
    "night": [55, 56, 58, 66, 67],
    "city": [30, 56, 58, 67, 68],
    "cyberpunk": [55, 57, 68, 69],
    "street": [56, 58, 69, 70],
    "rain": [57, 58, 70, 71],
    "dark": [56, 58, 67, 71],
    "coffee": [60, 61, 62, 72, 73],
    "cup": [60, 61, 72, 73],
    "mug": [60, 61, 72, 73],
    "espresso": [61, 62, 73, 74],
    "latte": [61, 62, 73, 74],
    "beans": [62, 74, 75],
    "camera": [63, 64, 76, 77],
    "vintage": [63, 64, 76, 77],
    "lens": [64, 76, 78],
    "retro": [63, 64, 77, 78],
    "photography": [64, 76, 78],

    # Colors & Aesthetic tones (coordinates 80 - 100)
    "golden": [80, 81],
    "orange": [81, 82],
    "warm": [80, 82],
    "blue": [83, 84],
    "cyan": [84, 85],
    "turquoise": [84, 85],
    "green": [86, 87],
    "emerald": [86, 87],
    "white": [88, 89],
    "bright": [88, 89],
    "black": [90, 91],
    "gray": [91, 92],
    "red": [93, 94],
    "crimson": [93, 94],
    "magenta": [95, 96],
    "pink": [95, 96],
    "purple": [96, 97],
    "yellow": [98, 99],
    "amber": [81, 99],

    # Categories (coordinates 101 - 110)
    "nature": [101, 102],
    "architecture": [103, 104],
    "urban": [105, 106],
    "objects": [107, 108],
    "technology": [107, 109],
    "lifestyle": [108, 110]
}

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Projects vector onto unit hypersphere ||v||_2 = 1.0"""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        # Avoid zero division
        v = np.ones_like(v) / np.sqrt(len(v))
        return v
    return v / norm

def text_to_embedding(text: str) -> list[float]:
    """
    Transforms natural language search query into normalized 128D continuous vector.
    """
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower()).strip()
    words = cleaned.split()

    if not words:
        # Uniform fallback
        return (np.ones(EMBEDDING_DIM) / np.sqrt(EMBEDDING_DIM)).tolist()

    matched_keywords = 0
    for word in words:
        # Direct concept space matching
        if word in CONCEPT_SPACE:
            for idx in CONCEPT_SPACE[word]:
                vec[idx] += 1.5
            matched_keywords += 1
        else:
            # Check substrings or lemma
            found = False
            for concept, indices in CONCEPT_SPACE.items():
                if concept in word or word in concept:
                    for idx in indices:
                        vec[idx] += 1.0
                    found = True
                    matched_keywords += 1
                    break
            
            # Deterministic word feature hashing for out-of-vocabulary terms into [111 - 127]
            if not found:
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                idx = 111 + (h % (EMBEDDING_DIM - 111))
                weight = 0.5 + ((h >> 8) % 100) / 200.0
                vec[idx] += weight

    # Add subtle base dispersion to avoid extreme sparsity
    vec += 0.02 * np.ones(EMBEDDING_DIM, dtype=np.float32)
    vec = normalize_vector(vec)
    return vec.tolist()

def image_to_embedding(image_path: str, metadata: dict = None) -> list[float]:
    """
    Extracts high-dimensional multimodal embedding from visual pixel contents + metadata.
    Analyzes spatial color layout, dominant palette, luminance, and texture.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    img = Image.open(image_path).convert('RGB')
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # 1. Visual Feature Extraction: Spatial Grid & Color Distribution
    small_img = img.resize((64, 64))
    arr = np.array(small_img, dtype=np.float32) / 255.0  # (64, 64, 3)

    r_mean = np.mean(arr[:, :, 0])
    g_mean = np.mean(arr[:, :, 1])
    b_mean = np.mean(arr[:, :, 2])
    lum = 0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean

    # Spatial halves (top sky/lighting vs bottom ground/surfaces)
    top_half = arr[:32, :, :]
    bottom_half = arr[32:, :, :]
    top_r, top_g, top_b = np.mean(top_half, axis=(0, 1))
    bot_r, bot_g, bot_b = np.mean(bottom_half, axis=(0, 1))

    # Color Profile Mapping to Visual Coordinates [80 - 100]
    # Golden / Orange (high R, medium G, low B)
    if r_mean > 0.4 and g_mean > 0.25 and b_mean < 0.4:
        warmth = (r_mean - b_mean) * 2.0
        vec[80] += warmth * 1.5
        vec[81] += warmth * 1.8
        vec[82] += warmth * 1.2
        vec[0] += warmth * 1.0  # sunset correlation
        vec[10] += warmth * 0.8 # desert / warm sand

    # Blue / Cyan (high B, medium G)
    if b_mean > 0.35 and b_mean >= r_mean:
        blue_strength = (b_mean - r_mean) * 2.5
        vec[83] += blue_strength * 2.0
        vec[84] += blue_strength * 1.8
        vec[2] += blue_strength * 1.2 # ocean / beach
        vec[31] += blue_strength * 1.0 # skyscraper glass

    # Green / Emerald (high G)
    if g_mean > r_mean and g_mean > b_mean:
        green_strength = (g_mean - min(r_mean, b_mean)) * 2.5
        vec[86] += green_strength * 2.0
        vec[87] += green_strength * 1.8
        vec[8] += green_strength * 1.5 # forest
        vec[20] += green_strength * 1.2 # trees

    # White / Crisp light (high luminance, balanced RGB)
    if lum > 0.65:
        white_strength = (lum - 0.65) * 3.0
        vec[88] += white_strength * 2.0
        vec[89] += white_strength * 1.5
        vec[6] += white_strength * 1.2 # snow
        vec[38] += white_strength * 1.0 # minimal interior

    # Dark / Midnight (low luminance)
    if lum < 0.25:
        dark_strength = (0.25 - lum) * 3.0
        vec[90] += dark_strength * 1.8
        vec[55] += dark_strength * 1.5 # neon night
        vec[56] += dark_strength * 1.5 # city night

    # Red / Crimson / Magenta
    if r_mean > 0.5 and g_mean < 0.3:
        red_strength = (r_mean - g_mean) * 2.0
        vec[93] += red_strength * 2.0
        vec[94] += red_strength * 1.5
        vec[14] += red_strength * 1.2 # autumn

    # Spatial composition: Top Blue + Bottom Brown/Sand (Beach / Desert)
    if top_b > top_r and bot_r > bot_b:
        vec[1] += 1.2 # beach
        vec[24] += 1.0 # dunes

    # Spatial composition: Top Sky + Bottom Dark (Urban night or Forest)
    if top_b > 0.4 and bot_g > 0.3:
        vec[8] += 1.0 # forest

    # 2. Texture & Edge Variance (Sobel gradient approximation)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    dy = np.abs(gray[1:, :] - gray[:-1, :])
    edge_density = float(np.mean(dx) + np.mean(dy))
    
    # High edge density = architecture/city/camera; Low edge density = sky/sand
    vec[30] += edge_density * 2.0 # architecture
    vec[63] += edge_density * 1.5 # camera

    # 3. Incorporate metadata tags & category (if provided during indexing)
    if metadata:
        category = metadata.get("category", "").lower()
        if category in CONCEPT_SPACE:
            for idx in CONCEPT_SPACE[category]:
                vec[idx] += 2.0

        tags = metadata.get("tags", "")
        if isinstance(tags, list):
            tags = " ".join(tags)
        
        for word in re.findall(r'[a-zA-Z]+', tags.lower()):
            if word in CONCEPT_SPACE:
                for idx in CONCEPT_SPACE[word]:
                    vec[idx] += 1.5

        title = metadata.get("title", "")
        for word in re.findall(r'[a-zA-Z]+', title.lower()):
            if word in CONCEPT_SPACE:
                for idx in CONCEPT_SPACE[word]:
                    vec[idx] += 1.2

    # Add subtle dispersion and normalize
    vec += 0.01 * np.ones(EMBEDDING_DIM, dtype=np.float32)
    vec = normalize_vector(vec)
    return vec.tolist()
