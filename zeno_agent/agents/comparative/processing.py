import pandas as pd
from typing import Dict, Any, List
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
        return {"summary": "", "cagr": 0.0, "avg_price": 0.0, "total_quantity": 0.0}

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = df.dropna(subset=["quantity", "price"])
    if df.empty:
        return {"summary": "", "cagr": 0.0, "avg_price": 0.0, "total_quantity": 0.0}

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
        "avg_price": avg_price,
        "total_quantity": total_quantity,
        "years": yearly["year"].tolist(),
        "yearly_prices": yearly["price"].tolist(),
        "yearly_quantities": yearly["quantity"].tolist()
    }


def get_structured_summary(query: str) -> Dict[str, Dict[str, Any]]:
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
            results[country] = summary
        except Exception as e:
            results[country] = {"summary": "", "cagr": 0.0, "avg_price": 0.0, "total_quantity": 0.0}
            print(f"[Comparative] DB lookup failed for {country}: {e}")
    return results


def get_rag_evidence(query: str, top_k: int = 5) -> str:
    try:
        embedding = encode_query_to_vector(query)
        raw_results = query_rag_embeddings_semantic(embedding, top_k=top_k)
        return merge_rag_content(raw_results) if raw_results else ""
    except Exception:
        return ""


def generate_comparative_charts(entities: Dict[str, Any], structured_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate comparison charts for countries/commodities"""
    countries = entities.get("countries", [])
    commodity = entities.get("commodity", "")
    
    if not countries or len(countries) < 2:
        return []
    
    charts = []
    
    # Chart 1: Average Price Comparison
    try:
        prices = []
        valid_countries = []
        for country in countries:
            if country in structured_data and structured_data[country].get("avg_price", 0) > 0:
                prices.append(structured_data[country]["avg_price"])
                valid_countries.append(country)
        
        if valid_countries:
            price_chart = {
                "x": valid_countries,
                "y": prices,
                "title": f"{commodity.capitalize()} Average Price Comparison (KES/kg)",
                "chart_type": "bar"
            }
            charts.append(price_chart)
    except Exception as e:
        print(f"[Comparative] Price chart generation failed: {e}")
    
    # Chart 2: Total Export Quantity Comparison
    try:
        quantities = []
        valid_countries = []
        for country in countries:
            if country in structured_data and structured_data[country].get("total_quantity", 0) > 0:
                quantities.append(structured_data[country]["total_quantity"])
                valid_countries.append(country)
        
        if valid_countries:
            quantity_chart = {
                "x": valid_countries,
                "y": quantities,
                "title": f"{commodity.capitalize()} Total Export Quantity Comparison (units)",
                "chart_type": "bar"
            }
            charts.append(quantity_chart)
    except Exception as e:
        print(f"[Comparative] Quantity chart generation failed: {e}")
    
    # Chart 3: CAGR Comparison
    try:
        cagrs = []
        valid_countries = []
        for country in countries:
            if country in structured_data and structured_data[country].get("cagr", 0) != 0:
                cagrs.append(structured_data[country]["cagr"])
                valid_countries.append(country)
        
        if valid_countries:
            cagr_chart = {
                "x": valid_countries,
                "y": cagrs,
                "title": f"{commodity.capitalize()} Growth Rate (CAGR %)",
                "chart_type": "bar"
            }
            charts.append(cagr_chart)
    except Exception as e:
        print(f"[Comparative] CAGR chart generation failed: {e}")
    
    # Chart 4: Price Trend Over Time (if data available)
    try:
        if len(countries) >= 2:
            country1 = countries[0]
            country2 = countries[1]
            
            if (country1 in structured_data and len(structured_data[country1].get("years", [])) > 0 and
                country2 in structured_data and len(structured_data[country2].get("years", [])) > 0):
                
                years1 = [str(y) for y in structured_data[country1].get("years", [])]
                prices1 = structured_data[country1].get("yearly_prices", [])
                
                years2 = [str(y) for y in structured_data[country2].get("years", [])]
                prices2 = structured_data[country2].get("yearly_prices", [])
                
                # Use first country's years as baseline, align second country
                if years1 and prices1:
                    trend_chart = {
                        "x": years1,
                        "y": prices1,
                        "title": f"{commodity.capitalize()} Price Trend: {country1} vs {country2}",
                        "chart_type": "line"
                    }
                    charts.append(trend_chart)
    except Exception as e:
        print(f"[Comparative] Trend chart generation failed: {e}")
    
    return charts


def synthesize_comparative_analysis(
    query: str,
    structured_data: Dict[str, Dict[str, Any]],
    rag_text: str,
    entities: Dict[str, Any]
) -> str:
    has_db_data = any(v.get("summary") for v in structured_data.values() if isinstance(v, dict))

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
    for country, data in structured_data.items():
        if isinstance(data, dict) and data.get("summary", "").strip():
            summary = data["summary"]
            cagr = data.get("cagr", 0)
            context_lines.append(f"{country}: {summary} (CAGR: {cagr:.2f}%)")
        else:
            context_lines.append(f"{country}: No structured data in local database — supplement with expert knowledge.")

    context = "\n".join(context_lines)
    if rag_text:
        context += f"\n\nKnowledge base evidence:\n{rag_text}"

    prompt = f"""You are Dr. Zeno, Senior Economist at the East African Trade Institute. You are producing a comprehensive policy brief for Ministers of Trade, commodity traders, and development finance institutions, comparing agricultural trade performance across countries.

USER QUERY:
{query}

LOCAL DATABASE DATA:
{context}

Write a full comparative economic policy brief with the following sections as flowing paragraphs:

SECTION 1 — PRODUCTION AND EXPORT OVERVIEW
Compare production volumes, export values in USD, global market share, farmer numbers, and GDP/forex contribution for each country. Use the database data provided and supplement with your expert knowledge of East African trade.

SECTION 2 — PRICE DYNAMICS AND MARKET POSITIONING
Compare export prices, quality premiums, auction systems, certification rates, and pricing power in global markets.

SECTION 3 — COMPETITIVE STRENGTHS AND WEAKNESSES
Analyze cost structures, yields per hectare, logistics efficiency, government support programs, extension services, and export destination diversity.

SECTION 4 — TRADE POLICY AND INSTITUTIONAL FRAMEWORK
Compare export taxes, levies, marketing boards, cooperative membership rates, and trade agreement coverage (COMESA, EAC, bilateral).

SECTION 5 — GROWTH TRAJECTORY AND MARKET TRENDS
Provide CAGR figures from the data, identify growth drivers, inflection points, and multi-year trends.

SECTION 6 — RISKS AND BLIND SPOTS
Climate vulnerability, disease/pest pressure, EUDR compliance costs, currency depreciation risks, youth migration from farms, input supply chains.

SECTION 7 — POLICY RECOMMENDATIONS
4-5 specific actionable recommendations per country. Name programs, institutions, timelines, and budget estimates where possible.

WRITING STANDARDS:
- World Bank policy paper quality — specific data, figures, institution names, policy programs
- No bullet points, no markdown headers — flowing professional paragraphs
- 1000-1500 words total
- Each section minimum 120 words
- Include comparative performance metrics and rankings
- End with one headline recommendation a Minister could act on this week

BEGIN POLICY BRIEF:""".strip()

    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[{"parts": [{"text": prompt}]}],
            config={"max_output_tokens": 5000, "temperature": 0.15}
        )
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()
        return economist_web_answer(query, "comparative", entities.get("commodity", ""), " vs ".join(entities.get("countries", [])))
    except Exception as e:
        print(f"[Comparative] LLM failed: {e}")
        return economist_web_answer(query, "comparative", entities.get("commodity", ""), " vs ".join(entities.get("countries", [])))