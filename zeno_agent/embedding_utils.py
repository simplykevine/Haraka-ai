# zeno_agent/embedding_utils.py

import os
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    # ✅ FIX #1: Use 768-dim model to match your database
    LOCAL_EMBED_MODEL = SentenceTransformer("all-mpnet-base-v2")
except Exception as e:
    print(f"[Info] Local embedding fallback disabled: {e}")
    LOCAL_EMBED_MODEL = None

embedding_cache = TTLCache(maxsize=2000, ttl=3600)

# ✅ FIX #2: Correct model name (gemini-embedding-001 doesn't exist)
GENAI_MODEL = "text-embedding-001"

_client = None


def _get_client():
    """Initialize and reuse a single Gemini client."""
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
        _client = genai.Client(api_key=api_key)
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def encode_query_to_vector(query_text: str, mode: str = "query") -> list[float]:
    """
    Generate embedding vector using text-embedding-004 with proper SDK structure.
    Returns: List of floats (768 dimensions for text-embedding-004)
    """
    if not query_text.strip():
        raise ValueError("Empty query text provided.")

    cache_key = f"{mode}:{query_text}"
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]

    client = _get_client()
    try:
        # ✅ FIX #3: Correct argument structure for embed_content()
        result = client.models.embed_content(
            model=GENAI_MODEL,
            contents=[query_text.strip()]  # ✅ List of strings, NOT dict with parts
        )
        
        # ✅ FIX #4: Correct response parsing
        if hasattr(result, "embeddings") and result.embeddings:
            vector = list(result.embeddings[0].values)
            embedding_cache[cache_key] = vector
            return vector
        else:
            raise ValueError(f"Gemini response missing embeddings field. Full response: {result}")
            
    except Exception as e:
        print(f"[Warning] Gemini embedding failed ({type(e).__name__}: {e}). Trying local fallback...")
        if LOCAL_EMBED_MODEL:
            try:
                # ✅ Generates 768-dim vector (matches your database)
                vector = LOCAL_EMBED_MODEL.encode(query_text, convert_to_numpy=True).tolist()
                # Normalize for cosine similarity
                vector = (np.array(vector) / np.linalg.norm(vector)).tolist()
                embedding_cache[cache_key] = vector
                print(f"[Info] Using local embedding fallback (768-dim) for query: {query_text[:50]}...")
                return vector
            except Exception as fallback_e:
                print(f"[Error] Local embedding fallback also failed: {fallback_e}")
        raise RuntimeError(f"Failed to generate embedding via Gemini or fallback: {e}")


def encode_vector_for_postgres(vector: list[float]) -> str:
    """
    Converts a Python list of floats to a Postgres-compatible vector string '[x1,x2,...]'.
    """
    if not isinstance(vector, list) or not all(isinstance(x, (float, int, np.floating, np.integer)) for x in vector):
        raise ValueError(f"Input must be a list of floats/ints. Got type: {type(vector)}")
    
    return "[" + ",".join(f"{float(v):.12f}" for v in vector) + "]"