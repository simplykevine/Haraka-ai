import os
from google import genai
from typing import List, Dict, Any
import numpy as np

from zeno_agent.embedding_utils import encode_query_to_vector

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")

client = genai.Client(api_key=GOOGLE_API_KEY)

GENERATION_MODEL = "models/gemini-2.5-flash"


def extract_entities(query: str) -> Dict[str, Any]:
    """Extract countries and commodities from query"""
    country_names = ["kenya", "tanzania", "uganda", "rwanda", "ethiopia", "burundi", "somalia", "drc", "congo"]
    commodity_names = ["coffee", "maize", "tea", "beans", "wheat", "rice", "sugar", "cocoa", "cotton", "oil"]

    query_lower = query.lower()
    countries = [c.title() for c in country_names if c in query_lower]
    commodities = [c for c in commodity_names if c in query_lower]

    return {
        "countries": list(dict.fromkeys(countries[:3])),  # Remove duplicates, max 3
        "commodity": commodities[0] if commodities else "coffee"
    }


def calculate_cagr(start_value: float, end_value: float, periods: int) -> float:
    """Calculate Compound Annual Growth Rate"""
    if start_value <= 0 or end_value <= 0 or periods <= 0:
        return 0.0
    try:
        return ((end_value / start_value) ** (1 / periods) - 1) * 100
    except:
        return 0.0


def merge_rag_content(rag_results: List[Dict[str, str]]) -> str:
    """Merge RAG results, removing duplicates"""
    seen = set()
    merged_content = []
    for doc in rag_results:
        content = doc.get("content", "").strip()
        if len(content) < 20:
            continue
        key = content[:100]
        if key in seen:
            continue
        seen.add(key)
        merged_content.append(content)
    return " ".join(merged_content)


def normalize_chart_data(data: List[float]) -> List[float]:
    """Normalize data for chart visualization"""
    if not data or len(data) == 0:
        return []
    
    try:
        data_array = np.array(data, dtype=float)
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            data_array = np.clip(data_array, lower_bound, upper_bound)
        
        return data_array.tolist()
    except:
        return data