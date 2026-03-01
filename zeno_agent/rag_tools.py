from typing import List, Dict
from google import genai
import os
from dotenv import load_dotenv
from .embedding_utils import encode_query_to_vector
from .db_utils import query_rag_embeddings_semantic
from .web_search import search_web, format_web_results

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")

client = genai.Client(api_key=GOOGLE_API_KEY)

GENERATION_MODEL = "models/gemini-2.5-flash"


def summarize_chunk(chunk_text: str, user_query: str) -> str:
    prompt = f"""Summarize the following text in 1-2 sentences relevant to the query. Return only the summary.

Query: "{user_query}"
Text: \"\"\"{chunk_text}\"\"\"
""".strip()

    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[{"parts": [{"text": prompt}]}],
            config={"max_output_tokens": 256}
        )
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()
        return chunk_text
    except Exception as e:
        print(f"[Warning] Summarization failed: {e}")
        return chunk_text


def get_base_rag_results(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    if not query.strip():
        return []
    try:
        embedding = encode_query_to_vector(query)
        results = query_rag_embeddings_semantic(embedding, top_k=top_k)
        if not results:
            return []
        summarized_results = []
        for r in results:
            summary = summarize_chunk(r["content"], query)
            summarized_results.append({
                "content": summary,
                "source": r.get("source", "Unknown")
            })
        return summarized_results
    except Exception as e:
        print(f"[Warning] RAG query failed: {e}")
        return []


def ask_knowledgebase_with_context(query: str, file_context: str = "", top_k: int = 5) -> str:
    # 1. Local RAG
    base_results = get_base_rag_results(query, top_k)
    rag_content_list = [r["content"] for r in base_results]
    rag_content = "\n\n".join(rag_content_list)
    has_rag = bool(rag_content_list)
    has_file = bool(file_context and file_context.strip())

    # 2. Web search
    web_results = search_web(
        query=f"{query} East Africa Kenya 2025 2026",
        num_results=5
    )
    web_content = format_web_results(web_results)
    has_web = bool(web_content)

    print(f"[RAG] DB results: {len(rag_content_list)} | Web results: {len(web_results)} | File: {has_file}")

    # 3. Build context block
    context_block = ""
    if has_file:
        context_block += f"=== UPLOADED DOCUMENT ===\n{file_context}\n\n"
    if has_rag:
        context_block += f"=== LOCAL KNOWLEDGE BASE ===\n{rag_content}\n\n"
    if has_web:
        context_block += f"=== LIVE WEB SEARCH RESULTS ===\n{web_content}\n\n"

    if context_block:
        final_prompt = f"""You are Dr. Zeno, Senior Economist at the East African Trade Institute — an expert in East African agricultural trade, commodity markets, tariff policy, macroeconomic analysis, and regional economic integration.

A user has asked you a question. You have access to live web search results, a local knowledge base, and possibly an uploaded document. Use ALL of this information to give a comprehensive, expert-level answer.

USER QUERY:
{query}

CONTEXT:
{context_block}

INSTRUCTIONS:
- Read every web search result carefully and extract ALL specific facts, figures, percentages, dollar amounts, and policy details mentioned.
- Combine web results with knowledge base context and your own expert economic knowledge.
- Structure your answer with clear sections (e.g. Impact on Exports, Impact on Imports, Policy Response, Recommendations).
- Include specific numbers, statistics, and dates wherever available from the context.
- Write in a professional economic narrative style — no markdown headers, no bullet points, just clear structured paragraphs.
- Your answer should be as comprehensive and data-rich as Gemini's best response on this topic.
- End with a concrete policy recommendation for East African governments or traders.

ANSWER:""".strip()
    else:
        final_prompt = f"""You are Dr. Zeno, Senior Economist at the East African Trade Institute — an expert in East African agricultural trade, commodity markets, tariff policy, macroeconomic analysis, and regional economic integration.

A user has asked you a question. Answer from your deep expert knowledge with specific data, figures, historical context, and economic reasoning.

USER QUERY:
{query}

INSTRUCTIONS:
- Provide a comprehensive, data-rich answer covering all relevant dimensions of the topic.
- Include specific statistics, policy names, dollar figures, and percentages where you know them.
- Structure into clear paragraphs covering different aspects (exports, imports, policy, recommendations).
- Write in professional economic narrative style — no bullet points or markdown.
- Focus on East Africa, especially Kenya, and practical implications for policymakers and traders.
- End with a concrete actionable recommendation.

ANSWER:""".strip()

    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[{"parts": [{"text": final_prompt}]}],
            config={
                "max_output_tokens": 3000,
                "temperature": 0.2
            }
        )
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()
        return "Dr. Zeno was unable to generate a response. Please try again."
    except Exception as e:
        print(f"[Warning] Final synthesis failed: {e}")
        return f"Unable to generate a response at this time. Error: {str(e)}"