import os
from cachetools import TTLCache
import numpy as np

GENERATION_MODEL = "models/gemini-2.5-flash"

try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBED_MODEL = SentenceTransformer("all-mpnet-base-v2")
    print("[Info] Embedding model loaded: all-mpnet-base-v2 (768-dim)")
except Exception as e:
    LOCAL_EMBED_MODEL = None
    print(f"[CRITICAL] Embedding model failed to load: {e}")

embedding_cache = TTLCache(maxsize=2000, ttl=3600)


def encode_query_to_vector(query_text: str, mode: str = "query") -> list[float]:
    if not query_text.strip():
        raise ValueError("Empty query text provided.")

    cache_key = f"{mode}:{query_text}"
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]

    if LOCAL_EMBED_MODEL is None:
        raise RuntimeError("Embedding model is not available. Check sentence-transformers installation.")

    vector = LOCAL_EMBED_MODEL.encode(query_text, convert_to_numpy=True).tolist()
    vector = (np.array(vector) / np.linalg.norm(vector)).tolist()
    embedding_cache[cache_key] = vector
    return vector


def encode_vector_for_postgres(vector: list[float]) -> str:
    if not isinstance(vector, list) or not all(isinstance(x, (float, int, np.floating, np.integer)) for x in vector):
        raise ValueError(f"Input must be a list of floats/ints. Got type: {type(vector)}")
    return "[" + ",".join(f"{float(v):.12f}" for v in vector) + "]"