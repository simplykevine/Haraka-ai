import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()


def search_web(query: str, num_results: int = 4) -> List[Dict[str, str]]:
    """
    Search the web using SerpAPI.
    Returns list of {title, snippet, link} dicts.
    Falls back to empty list if API not configured or fails.
    """
    serpapi_key = os.getenv("SERPAPI_KEY")
    if not serpapi_key:
        print("[WebSearch] SERPAPI_KEY not set. Skipping web search.")
        return []

    try:
        from serpapi import GoogleSearch

        search = GoogleSearch({
            "q": query,
            "num": num_results,
            "api_key": serpapi_key,
        })
        results = search.get_dict()
        organic = results.get("organic_results", [])

        return [
            {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            }
            for item in organic[:num_results]
        ]
    except Exception as e:
        print(f"[WebSearch] SerpAPI search failed: {e}")
        return []


def format_web_results(results: List[Dict[str, str]]) -> str:
    """Format web results into a readable context string for the LLM."""
    if not results:
        return ""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[Web Source {i}] {r['title']}")
        lines.append(f"{r['snippet']}")
        lines.append(f"URL: {r['link']}")
        lines.append("")
    return "\n".join(lines)