import os
import re
from google import genai
from .scenario_db import build_structured_context
from .scenario_helpers import (
    merge_rag_content,
    encode_query_to_vector,
    query_rag_embeddings_semantic,
)
from zeno_agent.economist_fallback import economist_web_answer

GENERATION_MODEL = "models/gemini-2.5-flash"


class ScenarioSubAgent:

    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is required for ScenarioSubAgent.")
        self.client = genai.Client(api_key=api_key)

    def extract_entities(self, query):
        commodity_match = re.search(
            r"(maize|coffee|tea|oil|wheat|sugar|rice|beans|sorghum|cassava|cotton|tobacco)",
            query, re.IGNORECASE
        )
        country_match = re.search(
            r"(kenya|uganda|tanzania|ethiopia|rwanda|burundi|south sudan|somalia)",
            query, re.IGNORECASE
        )
        commodity = commodity_match.group(1).lower() if commodity_match else None
        country = country_match.group(1).lower() if country_match else None
        return commodity, country

    def get_rag_context(self, query):
        try:
            embedding = encode_query_to_vector(query)
            results = query_rag_embeddings_semantic(embedding, top_k=5)
            if results:
                return merge_rag_content(results)
        except Exception as e:
            print(f"[Scenario] RAG error: {e}")
        return ""

    def handle_with_context(self, scenario_query: str, file_context: str = "") -> dict:
        query = scenario_query.strip()
        commodity, country = self.extract_entities(query)

        if not commodity or not country:
            print("[Scenario] No entities found — using economist web fallback")
            analysis = economist_web_answer(
                query=query,
                agent_type="scenario",
                commodity=commodity or "",
                country=country or ""
            )
            return {
                "type": "scenario",
                "query": query,
                "entities": {"commodity": commodity, "country": country},
                "llm_analysis": analysis,
                "followup": "Try asking about a specific commodity and country for more targeted analysis."
            }

        # Always gather all available context
        structured_context = build_structured_context(commodity, country)
        rag_context = self.get_rag_context(query)

        # Build extra_context string to pass into fallback
        extra_parts = []
        if file_context:
            extra_parts.append(f"Uploaded document context:\n{file_context[:2000]}")
        if structured_context and structured_context != "No structured economic data available.":
            extra_parts.append(f"Local structured trade data for {country} {commodity}:\n{structured_context}")
        if rag_context:
            extra_parts.append(f"Local knowledge base evidence:\n{rag_context[:1000]}")

        extra_context = "\n\n".join(extra_parts)

        # Always use economist_web_answer for scenario — it runs 3 targeted web searches
        # and produces IMF-quality policy brief. Local data is passed as extra_context.
        print(f"[Scenario] Running economist_web_answer for {commodity}/{country}")
        analysis = economist_web_answer(
            query=query,
            agent_type="scenario",
            commodity=commodity,
            country=country,
            extra_context=extra_context if extra_context else ""
        )

        return {
            "type": "scenario",
            "query": query,
            "entities": {"commodity": commodity, "country": country},
            "llm_analysis": analysis,
            "followup": "Try adjusting policy parameters or macroeconomic assumptions for a deeper analysis."
        }
