import pandas as pd
from typing import Dict, Any
from .utils import (
    client,
    GENERATION_MODEL,
    extract_entities,
    calculate_cagr,
    merge_rag_content,
    encode_query_to_vector,
)
from zeno_agent.db_utils import (
    get_country_id_by_name,
    get_product_id_by_name,
    get_indicator_id_by_metric,
    get_trade_data_from_db,
    query_rag_embeddings_semantic,
)
from zeno_agent.economist_fallback import economist_web_answer


def summarize_trade_data(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"summary": "", "cagr": 0.0}

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = df.dropna(subset=["quantity", "price"])
    if df.empty:
        return {"summary": "", "cagr": 0.0}

    yearly = df.groupby("year").agg({"quantity": "sum", "price": "mean"}).reset_index()
    start_year, end_year = yearly["year"].min(), yearly["year"].max()
    total_quantity = yearly["quantity"].sum()
    avg_price = yearly["price"].mean()

    if len(yearly) > 1:
        cagr = calculate_cagr(
            float(yearly.iloc[0]["quantity"]), float(yearly.iloc[-1]["quantity"]), end_year - start_year
        )
    else:
        cagr = 0.0

    return {
        "summary": f"Period {start_year}-{end_year}: total {total_quantity:,} units, "
                   f"avg price KES {avg_price:,.2f}, CAGR {cagr:.2f}%",
        "cagr": cagr,
    }


def get_structured_summary(query: str) -> Dict[str, str]:
    entities = extract_entities(query)
    countries = entities["countries"]
    commodity = entities["commodity"]
    results = {}

    for country in countries:
        try:
            country_id = get_country_id_by_name(country)
            product_id = get_product_id_by_name(commodity)
            indicator_id = get_indicator_id_by_metric("exports")
            df = get_trade_data_from_db(country_id, product_id, indicator_id)
            summary = summarize_trade_data(df)
            results[country] = summary["summary"]
        except Exception as e:
            results[country] = ""
            print(f"[Comparative] DB lookup failed for {country}: {e}")
    return results


def get_rag_evidence(query: str, top_k: int = 5) -> str:
    try:
        embedding = encode_query_to_vector(query)
        raw_results = query_rag_embeddings_semantic(embedding, top_k=top_k)
        return merge_rag_content(raw_results) if raw_results else ""
    except Exception:
        return ""


def synthesize_comparative_analysis(
    query: str,
    structured_data: Dict[str, str],
    rag_text: str,
    entities: Dict[str, Any]
) -> str:
    has_db_data = any(v for v in structured_data.values() if v.strip())

    if not has_db_data and not rag_text.strip():
        print("[Comparative] No DB data — using economist web fallback")
        countries = " vs ".join(entities.get("countries", []))
        commodity = entities.get("commodity", "")
        return economist_web_answer(
            query=query,
            agent_type="comparative",
            commodity=commodity,
            country=countries
        )

    context_lines = []
    for country, summary in structured_data.items():
        if summary.strip():
            context_lines.append(f"{country}: {summary}")
        else:
            context_lines.append(f"{country}: No structured data in local database — supplement with expert knowledge.")

    context = "\n".join(context_lines)
    if rag_text:
        context += f"\n\nKnowledge base evidence:\n{rag_text}"

    prompt = f"""You are Dr. Zeno, Senior Economist at the East African Trade Institute. You are producing a comprehensive policy brief for Ministers of Trade, commodity traders, and development finance institutions.

USER QUERY:
{query}

LOCAL DATABASE DATA:
{context}

Write a full comparative economic policy brief with the following sections as flowing paragraphs:

SECTION 1 — PRODUCTION AND EXPORT OVERVIEW
Compare production volumes, export values in USD, global market share, farmer numbers, and GDP/forex contribution for each country. Use the database data provided and supplement with your expert knowledge where data is missing.

SECTION 2 — PRICE DYNAMICS AND MARKET POSITIONING
Compare export prices, quality premiums, auction systems, certification, and pricing power in global markets.

SECTION 3 — COMPETITIVE STRENGTHS AND WEAKNESSES
Analyze cost structures, yields, logistics, government support, and export destination diversity.

SECTION 4 — TRADE POLICY AND INSTITUTIONAL FRAMEWORK
Compare export taxes, levies, marketing boards, cooperative systems, and trade agreement coverage.

SECTION 5 — GROWTH TRAJECTORY
Provide CAGR figures from the data, identify growth drivers and inflection points.

SECTION 6 — RISKS AND BLIND SPOTS
Climate vulnerability, disease pressure, EUDR compliance, currency risks, youth migration from farms.

SECTION 7 — POLICY RECOMMENDATIONS
4-5 specific actionable recommendations per country. Name programs, institutions, and timelines.

WRITING STANDARDS:
- World Bank policy paper quality — specific data, figures, institution names, policy names
- No bullet points, no markdown headers — flowing professional paragraphs
- 800-1200 words total
- Each section minimum 100 words
- End with one headline recommendation a Minister could act on this week

BEGIN POLICY BRIEF:""".strip()

    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[{"parts": [{"text": prompt}]}],
            config={"max_output_tokens": 4000, "temperature": 0.15}
        )
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()
        return economist_web_answer(query, "comparative", entities.get("commodity", ""), " vs ".join(entities.get("countries", [])))
    except Exception as e:
        print(f"[Comparative] LLM failed: {e}")
        return economist_web_answer(query, "comparative", entities.get("commodity", ""), " vs ".join(entities.get("countries", [])))