import os
import re
from google import genai
from .scenario_db import build_structured_context
from .scenario_helpers import (
    merge_rag_content,
    encode_query_to_vector,
    build_scenario_prompt,
    query_rag_embeddings_semantic,
)


class ScenarioSubAgent:
    """Economist-focused LLM-powered scenario analysis agent"""

    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is required for ScenarioSubAgent.")
        self.client = genai.Client(api_key=api_key)

    def extract_entities(self, query):
        """Extract commodity and country from the query text."""
        commodity_match = re.search(r"(maize|coffee|tea|oil|wheat|sugar)", query, re.IGNORECASE)
        country_match = re.search(r"(kenya|uganda|tanzania|ethiopia|rwanda)", query, re.IGNORECASE)

        commodity = commodity_match.group(1).lower() if commodity_match else None
        country = country_match.group(1).lower() if country_match else None
        return commodity, country

    def get_rag_context(self, query):
        """Retrieve relevant context from RAG documents."""
        try:
            embedding = encode_query_to_vector(query)
            results = query_rag_embeddings_semantic(embedding, top_k=5)
            if results:
                return merge_rag_content(results)
        except Exception as e:
            print(f"[Scenario] RAG error: {e}")
        return "No relevant policy, macroeconomic, or event documents found."

    def handle_with_context(self, scenario_query: str, file_context: str = "") -> dict:
        query = scenario_query.strip()
        commodity, country = self.extract_entities(query)

        if not commodity or not country:
            return {
                "type": "scenario",
                "query": query,
                "response": "Please specify a commodity (e.g., coffee, maize) and country (e.g., Kenya, Rwanda).",
                "followup": "Example: 'What if Kenya subsidizes coffee production?'",
            }

        structured_context = build_structured_context(commodity, country)
        rag_context = self.get_rag_context(query)
    

        context_parts = []
        if file_context:
            context_parts.append(f"Uploaded document:\n{file_context}")
        if structured_context:
            context_parts.append(f"Structured Economic Data: {structured_context}")
        if rag_context:
            context_parts.append(f"Policy/Documents: {rag_context}")
    
        full_context = "\n\n".join(context_parts)

        prompt = f"""
You are Dr. Zeno, Senior Economist at the East African Trade Institute.
Write a detailed professional plain text economic analysis.

Scenario Query: "{query}"

Available Context:
{full_context}

Instructions:
- Use clear, formal economic language.
- Organize into short, readable paragraphs (no markdown).
- Discuss: Immediate Effects, Price Dynamics, Trade Effects, Macroeconomic Implications, Risks.
- Base reasoning on the data provided.
- Keep it 250-300 words.
- Output plain text only.
""".strip()  # ✅ Critical: Remove leading/trailing whitespace

        try:
            # ✅ CRITICAL FIX #1: Correct model name for YOUR endpoint
            # ✅ CRITICAL FIX #2: Proper content structure required by google.genai SDK
            response = self.client.models.generate_content(
                model="gemini-flash-latest",  # ✅ NOT models/gemini-1.5-flash-002
                contents=[{"parts": [{"text": prompt}]}]  # ✅ Required structure (NOT raw string)
            )
            
            # ✅ CRITICAL FIX #3: Correct response parsing for new SDK
            if response.candidates and response.candidates[0].content.parts:
                analysis = response.candidates[0].content.parts[0].text.strip()
            else:
                analysis = (
                    f"Scenario analysis for {commodity} in {country.title()} requires additional context. "
                    f"Consider providing specific policy parameters or economic assumptions for a detailed assessment."
                )
                
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)[:200]  # Truncate long errors
            print(f"[ERROR] Scenario analysis failed ({error_type}): {error_msg}")
            analysis = (
                f"A {commodity} policy scenario in {country.title()} would affect production and trade flows. "
                f"Without specific parameters, general effects include: supply elasticity impacts on domestic prices, "
                f"export competitiveness changes, and fiscal implications for government budgets. "
                f"Provide specific policy details for a tailored analysis."
            )

        return {
            "type": "scenario",
            "query": query,
            "entities": {"commodity": commodity, "country": country},
            "llm_analysis": analysis,
            "followup": "Try adjusting policy parameters or macroeconomic assumptions."
        }