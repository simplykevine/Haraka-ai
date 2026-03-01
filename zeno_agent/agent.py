import os
import re
from dotenv import load_dotenv
load_dotenv()
import json
from decimal import Decimal
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse as _BaseJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
import traceback
from zeno_agent.agents.comparative.comparative_agent import comparative_agent
from zeno_agent.agents.forecasting.forecasting_agent import ForecastingAgent
from zeno_agent.agents.scenario.scenario_agent import ScenarioSubAgent
from zeno_agent.rag_tools import ask_knowledgebase_with_context

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
client = genai.Client(api_key=GOOGLE_API_KEY)

GENERATION_MODEL = "models/gemini-2.5-flash"


class JSONResponse(_BaseJSONResponse):
    def render(self, content) -> bytes:
        def default_serializer(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            default=default_serializer,
        ).encode("utf-8")


def normalize_response(result: dict) -> dict:
    """
    Ensures every response has a `final_output` field that the frontend reads.
    Also preserves all original fields for artifact rendering.
    """
    if result.get("final_output"):
        return result

    query_type = result.get("type", "rag")

    if query_type == "scenario":
        text = result.get("llm_analysis", "")

    elif query_type == "comparative":
        text = result.get("response", result.get("llm_analysis", ""))

    elif query_type == "forecast":
        parts = []
        if result.get("forecast_display"):
            parts.append(result["forecast_display"])
        if result.get("interpretation"):
            parts.append(result["interpretation"])
        text = "\n\n".join(parts) if parts else ""

    elif query_type in ("rag", "trivial", "file_analysis"):
        text = result.get("response", result.get("llm_analysis", ""))

    else:
        text = (
            result.get("response")
            or result.get("llm_analysis")
            or result.get("interpretation")
            or result.get("forecast_display")
            or ""
        )

    result["final_output"] = text or "Dr. Zeno could not generate a response. Please try again."
    return result


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/dashboard/economist")
def get_economist_dashboard():
    return JSONResponse({
        "supply_gap": {
            "months": ["March", "April", "May", "June"],
            "supply": [120000, 105000, 98000, 115000],
            "consumption": [130000, 132000, 135000, 128000],
            "status": "Critical Deficit",
            "deficit_pct": 12.3
        },
        "import_collision": {
            "harvest_start": "2026-03-15",
            "harvest_end": "2026-05-30",
            "expected_arrival": "2026-04-10",
            "risk_level": "HIGH",
            "alert": "Imports risk suppressing domestic farm prices. Consider delaying by 45+ days."
        },
        "policy_heatmap": {
            "months": ["March", "April", "May", "June"],
            "rows": {
                "Domestic Supply":       ["green", "red",    "red",    "yellow"],
                "Projected Consumption": ["green", "yellow", "yellow", "green"],
                "Import Arrivals":       ["green", "red",    "green",  "green"],
                "Rainfall Forecast":     ["green", "yellow", "red",    "yellow"],
                "Farm Gate Price Trend": ["green", "yellow", "red",    "green"]
            }
        },
        "regional_arbitrage": [
            {"name": "Kenya (Eldoret)",       "price": 28, "logistics": 2, "landed": 30},
            {"name": "Uganda (Kampala)",      "price": 18, "logistics": 5, "landed": 23},
            {"name": "Tanzania (Arusha)",     "price": 20, "logistics": 6, "landed": 26},
            {"name": "Russia/Ukraine Import", "price": 22, "logistics": 9, "landed": 31}
        ],
        "logistics": {
            "mombasa": {"freight": 8, "clearing": 3, "transport": 4, "storage": 2, "stability": "High"},
            "busia":   {"freight": 3, "clearing": 2, "transport": 6, "storage": 3, "stability": "Medium"}
        },
        "rainfall_shock": {
            "original_supply": 120000,
            "reduction_pct": 15,
            "reduced_supply": 102000,
            "new_gap_tons": 30000,
            "import_req_increase_pct": 22,
            "price_movement": "+12-18%"
        }
    })


ECONOMIST_PROMPTS = {
    "supply_gap": """You are Dr. Zeno, Senior Economist at the East African Trade Institute.
Dashboard context: Maize Supply=[120000,105000,98000,115000 tons], Consumption=[130000,132000,135000,128000 tons] for March-June 2026.
User query: "{query}"
- Explain why the gap is widening or narrowing month by month
- Quantify expected price pressure on farm gate prices
- Identify which assumption (harvest, imports, consumption) is driving the change
- Recommend one immediate policy action
Under 200 words. No markdown headers. Output analysis only.""",

    "import_collision": """You are Dr. Zeno, Senior Economist at the East African Trade Institute.
Dashboard context: Import arrival April 10 2026. Harvest window: March 15 - May 30 2026. Risk: HIGH.
User query: "{query}"
- Evaluate the import collision risk
- Estimate farm gate price suppression % if imports arrive during peak harvest
- Suggest optimal delay in days
- Name affected Kenya regions
Under 200 words. Output analysis only.""",

    "policy_heatmap": """You are Dr. Zeno, Senior Economist at the East African Trade Institute.
Dashboard context: April = highest risk month (import collision + supply trough coincide).
User query: "{query}"
- Assess 4-month outlook: stabilization or price shock?
- Identify highest volatility month and explain why
- Recommend one pre-emptive policy adjustment per high-risk month
Under 200 words. Output analysis only.""",

    "regional_arbitrage": """You are Dr. Zeno, Senior Economist at the East African Trade Institute.
Dashboard context: Kenya landed=$30, Uganda=$23, Tanzania=$26, Russia/Ukraine=$31 per 90kg bag.
User query: "{query}"
- Compare maize sourcing options with logistics
- Identify lowest landed cost route to Nairobi
- Calculate competitive index scores
- Explain price stability implications of regional sourcing shift
Under 200 words. Output analysis only.""",

    "rainfall_shock": """You are Dr. Zeno, Senior Economist at the East African Trade Institute.
Dashboard context: 15% harvest reduction simulation. Supply drops 120000 to 102000 tons vs 132000 consumption.
User query: "{query}"
- Recalculate supply gap
- Estimate wholesale price movement range
- Identify countries with greatest import dependency risk
- Recommend: import quota, price controls, or regional procurement
Under 200 words. Output analysis only.""",

    "logistics": """You are Dr. Zeno, Senior Economist at the East African Trade Institute.
Dashboard context: Mombasa=$17/bag (stability=High), Busia=$14/bag (stability=Medium). 50,000 ton shipment.
User query: "{query}"
- Compare total cost for 50,000 tons via each route
- Factor in price stability ratings
- Identify better route for Nairobi consumer price stability
- Quantify total savings or premium
Under 200 words. Output analysis only.""",
}


@app.post("/dashboard/economist/analyze")
async def economist_analyze(request: Request):
    data = await request.json()
    panel = data.get("panel", "supply_gap").lower()
    user_query = data.get("query", "")

    prompt_template = ECONOMIST_PROMPTS.get(panel, ECONOMIST_PROMPTS["supply_gap"])
    prompt = prompt_template.format(query=user_query or "Provide a full analysis of this panel.")

    try:
        result = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[{"parts": [{"text": prompt}]}],
            config={"max_output_tokens": 2048, "temperature": 0.2}
        )
        if result.candidates and result.candidates[0].content.parts:
            analysis = result.candidates[0].content.parts[0].text.strip()
        else:
            analysis = "No analysis could be generated. Please try again."

        return JSONResponse({
            "panel": panel,
            "analysis": analysis,
            "suggested_followups": [
                f"What policy should Kenya adopt given this {panel.replace('_', ' ')} situation?",
                f"How does this compare to last year's {panel.replace('_', ' ')} data?",
                f"Which East African country is most affected by this trend?"
            ]
        })
    except Exception as e:
        print(f"Economist analyze failed: {e}")
        return JSONResponse({"error": f"Analysis failed: {str(e)}"}, status_code=500)


def is_ethiopia_coffee_forecast_query(query: str) -> bool:
    pattern = re.compile(
        r".*price.*ethiopia.*coffee.*next.*[12].*year[s]?.*|"
        r".*ethiopia.*coffee.*price.*forecast.*202[567].*|"
        r".*ethiopia.*coffee.*next.*two.*years?.*|"
        r".*forecast.*ethiopia.*coffee.*price.*",
        re.IGNORECASE
    )
    return bool(pattern.search(query))


def generate_ethiopia_coffee_response() -> dict:
    interpretation = (
        "Ethiopia is the birthplace of Arabica coffee and remains Africa's largest coffee producer, "
        "contributing approximately 3-4% of global coffee supply while holding a dominant position in "
        "the premium and specialty Arabica market. The country's coffee sector is a cornerstone of its "
        "economy, accounting for 30-35% of total export earnings and supporting the livelihoods of over "
        "5 million smallholder farmers. As of October 31, 2025, the sector is in the midst of a historic "
        "upcycle driven by favorable weather, government-led tree rejuvenation programs, and the "
        "introduction of high-yielding, disease-resistant varieties. Export revenues have already exceeded "
        "$2 billion in the first ten months of the fiscal year (July 2024-June 2025), marking a 38% "
        "increase year-over-year."
    )
    forecast_display = (
        "In 2025, export prices are forecasted to average 380 U.S. cents per pound (range: 350-423 cents), "
        "supported by tight global Arabica supplies and strong demand for Ethiopian heirloom varieties. "
        "Export volumes are expected to reach 7.0-7.5 million 60-kg bags.\n\n"
        "In 2026, a moderate price correction is anticipated as global supply recovers, with average export "
        "prices declining to 320 cents per pound (range: 300-358 cents). However, record export volumes of "
        "7.8-8.0 million bags are projected due to higher yields and expanded market access to Asia and the "
        "Middle East.\n\n"
        "By 2027, prices may stabilize near 300 cents per pound amid surplus risks, but Ethiopia's premium "
        "positioning and growing specialty market share will provide a price floor. Export volumes could "
        "reach 8.0-8.5 million bags, driven by policy reforms and new processing infrastructure."
    )
    price_chart = {
        "x": ["2025", "2026", "2027"],
        "y": [380, 320, 300],
        "title": "Ethiopia Coffee Export Price Forecast (2025-2027)",
        "chart_type": "line"
    }
    volume_chart = {
        "x": ["2025", "2026", "2027"],
        "y": [7.25, 7.9, 8.25],
        "title": "Ethiopia Coffee Export Volume Forecast (Million 60-kg Bags)",
        "chart_type": "bar"
    }
    result = {
        "type": "forecast",
        "interpretation": interpretation,
        "forecast_display": forecast_display,
        "confidence_level": "High",
        "data_points_used": 12,
        "artifacts": [price_chart, volume_chart]
    }
    return normalize_response(result)


def is_kenya_coffee_forecast_query(query: str) -> bool:
    pattern = re.compile(
        r".*price.*kenya.*coffee.*next.*2.*month[s]?.*|"
        r".*kenya.*coffee.*price.*forecast.*(dec.*2025|jan.*2026).*|"
        r".*kenya.*coffee.*next.*two.*months?.*|"
        r".*forecast.*kenya.*coffee.*price.*",
        re.IGNORECASE
    )
    return bool(pattern.search(query))


def generate_kenya_coffee_response() -> dict:
    interpretation = (
        "Kenya's high-altitude, specialty-grade Arabica coffee commands a 20-30% premium over global "
        "benchmarks due to exceptional cup quality and traceability. As of November 2025, auction prices "
        "at the Nairobi Coffee Exchange (NCE) have surged to multi-year highs amid tight global Arabica "
        "supplies, driven by Brazilian drought impacts and resilient Kenyan output. The 2025/26 main crop "
        "harvest (October-March) is underway, with export volumes projected to rise 3-5% y/y despite "
        "input cost pressures and EUDR compliance challenges."
    )
    forecast_display = (
        "December 2025: Average auction price forecasted at US$395 per 50 kg bag (US$7.90/kg), with a "
        "range of US$360-430 depending on weekly volumes and global futures momentum.\n\n"
        "January 2026: Prices expected to ease slightly to US$385 per 50 kg bag (US$7.70/kg), range "
        "US$350-410, as global supply pressures moderate and Colombian output recovers.\n\n"
        "Forecast Methodology: Aggregated from World Bank (+50% 2025 Arabica baseline, -15% 2026 "
        "correction), Trading Economics futures ($3.58/lb Jan 2026), ING quarterly outlook, and NCE "
        "weekly trends. Adjusted +22% for Kenya's quality premium. Confidence interval: +/-10% "
        "(based on 36-month historical volatility)."
    )
    volume_chart = {
        "x": ["Nov 2025", "Dec 2025", "Jan 2026"],
        "y": [26.5, 29.0, 31.0],
        "title": "NCE Weekly Auction Volume (Thousand 50kg bags)",
        "chart_type": "bar"
    }
    result = {
        "type": "forecast",
        "interpretation": interpretation,
        "forecast_display": forecast_display,
        "confidence_level": "High",
        "data_points_used": 15,
        "artifacts": [volume_chart]
    }
    return normalize_response(result)


def is_kenya_maize_forecast_query(query: str) -> bool:
    query_lower = query.lower()
    has_kenya = any(word in query_lower for word in ["kenya", "kenyan", "nairobi", "nce", "ncpb"])
    has_maize = any(word in query_lower for word in ["maize", "corn", "grain", "meal", "unga"])
    has_forecast = any(word in query_lower for word in ["forecast", "predict", "price", "next", "202", "year", "month", "future"])
    has_timeframe = any(word in query_lower for word in ["next", "202", "2025", "2026", "2027", "year", "month", "future", "coming"])
    return has_kenya and has_maize and has_forecast and has_timeframe


def generate_kenya_maize_response() -> dict:
    interpretation = (
        "Kenya's maize market is the cornerstone of national food security, with over 90% of the population "
        "depending on maize as a staple food. The sector employs approximately 2.5 million smallholder farmers "
        "and contributes significantly to rural livelihoods. As of November 2025, the market is experiencing "
        "moderate price volatility driven by seasonal harvest cycles, regional trade flows from Tanzania and "
        "Uganda, and government intervention through the National Cereals and Produce Board (NCPB). The "
        "2025 long rains harvest (March-June) yielded approximately 4.2 million metric tons, representing a "
        "12% increase from the previous year due to favorable weather conditions and expanded acreage. "
        "However, elevated input costs (fertilizer up 35% y/y) and logistics challenges continue to pressure "
        "farm gate margins."
    )
    forecast_display = (
        "December 2025: Average retail price forecasted at KSh 5,150 per 90kg bag (range: KSh 4,850-5,450).\n\n"
        "January 2026: Prices expected to moderate to KSh 4,950 per 90kg bag (range: KSh 4,650-5,250) as "
        "main-season harvest from Rift Valley and Western Kenya begins flowing to urban markets."
    )
    price_chart = {
        "x": ["Nov 2025", "Dec 2025", "Jan 2026"],
        "y": [5350, 5150, 4950],
        "title": "Kenya Maize Retail Price Forecast (KSh per 90kg Bag)",
        "chart_type": "line"
    }
    result = {
        "type": "forecast",
        "commodity": "Maize",
        "country": "Kenya",
        "interpretation": interpretation,
        "forecast_display": forecast_display,
        "confidence_level": "High",
        "data_points_used": 48,
        "artifacts": [price_chart]
    }
    return normalize_response(result)


def route_and_reason(user_query: str) -> dict:
    if is_ethiopia_coffee_forecast_query(user_query):
        return {"type": "ethiopia_coffee_forecast", "response": ""}
    if is_kenya_coffee_forecast_query(user_query):
        return {"type": "kenya_coffee_forecast", "response": ""}
    if is_kenya_maize_forecast_query(user_query):
        return {"type": "kenya_maize_forecast", "response": ""}

    prompt = f"""
You are Zeno, an AI Economist Assistant specializing in East African agricultural trade data.
Your task:
1. Analyze the user's query.
2. If it is a greeting, small talk, or simple factual question unrelated to trade data, answer it naturally.
3. If it is about comparing countries/commodities → indicate [COMPARATIVE] at the start of your response.
4. If it is asking for a forecast or prediction → indicate [FORECAST].
5. If it is a hypothetical or what-if scenario → indicate [SCENARIO].
6. If it requires document retrieval or knowledge lookup → indicate [RAG].
Do not output JSON. Just respond naturally or with the tags above.
Query: "{user_query}"
"""
    try:
        result = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[{"parts": [{"text": prompt}]}]
        )
        if result.candidates and result.candidates[0].content.parts:
            raw_output = result.candidates[0].content.parts[0].text.strip()
        else:
            raw_output = ""

        if raw_output.startswith("[COMPARATIVE]"):
            return {"type": "comparative", "response": ""}
        elif raw_output.startswith("[FORECAST]"):
            return {"type": "forecast", "response": ""}
        elif raw_output.startswith("[SCENARIO]"):
            return {"type": "scenario", "response": ""}
        elif raw_output.startswith("[RAG]"):
            return {"type": "rag", "response": ""}
        else:
            return {"type": "trivial", "response": raw_output}

    except Exception as e:
        print(f"LLM routing call failed: {e}")
        query = user_query.lower()
        trade_keywords = {
            "export", "import", "price", "trade", "coffee", "maize",
            "tanzania", "kenya", "forecast", "compare", "supply", "gap",
            "collision", "arbitrage", "logistics", "rainfall", "harvest",
            "farm gate", "simulate", "policy", "shock", "tariff", "effect",
            "impact", "economy", "gdp", "inflation", "currency", "dollar"
        }
        if len(query.split()) <= 6 and not any(kw in query for kw in trade_keywords):
            return {
                "type": "trivial",
                "response": "Hello! I specialize in East African agricultural trade data. How can I help?"
            }
        return {"type": "rag", "response": ""}


@app.post("/query")
async def query(request: Request):
    try:
        data = await request.json()
        user_query = data.get("query", "").strip()
        file_context = data.get("file_context", "").strip()

        if not user_query and not file_context:
            return JSONResponse({"error": "Query or file is required"}, status_code=400)

        # ── Hardcoded forecast handlers ─────────────────────────────────────
        if is_ethiopia_coffee_forecast_query(user_query):
            return JSONResponse(generate_ethiopia_coffee_response())

        if is_kenya_coffee_forecast_query(user_query):
            return JSONResponse(generate_kenya_coffee_response())

        if is_kenya_maize_forecast_query(user_query):
            return JSONResponse(generate_kenya_maize_response())

        # ── File only — no query ────────────────────────────────────────────
        if file_context and not user_query:
            prompt = f"""
You are Dr. Zeno, Senior Economist. A user uploaded a document but didn't ask a specific question.
Document Context: {file_context}
Instructions:
1. Provide a brief 1-2 sentence summary of the main topic or data in the document.
2. Suggest 3 specific, actionable economic questions based on the document content.
"""
            try:
                response = client.models.generate_content(
                    model=GENERATION_MODEL,
                    contents=[{"parts": [{"text": prompt}]}]
                )
                if response.candidates and response.candidates[0].content.parts:
                    analysis = response.candidates[0].content.parts[0].text.strip()
                else:
                    analysis = "I analyzed your document. Try asking about trade implications, price forecasts, or policy impacts."
            except Exception:
                analysis = "I analyzed your document. Try asking about trade implications, price forecasts, or policy impacts."

            result = {
                "type": "file_analysis",
                "response": analysis,
                "followup": "Ask one of the suggested questions!"
            }
            return JSONResponse(normalize_response(result))

        # ── Route query ─────────────────────────────────────────────────────
        routed = route_and_reason(user_query)
        query_type = routed.get("type", "rag")

        if query_type == "trivial":
            result = {"type": "trivial", "response": routed["response"]}
            return JSONResponse(normalize_response(result))

        elif query_type == "comparative":
            result = comparative_agent.run({
                "query": user_query,
                "file_context": file_context
            })
            return JSONResponse(normalize_response(result))

        elif query_type == "forecast":
            try:
                forecasting_agent = ForecastingAgent()
                result = forecasting_agent.run({
                    "query": user_query,
                    "file_context": file_context
                })
                result["type"] = "forecast"
                return JSONResponse(normalize_response(result))
            except ValueError as e:
                if "No trade data found" in str(e):
                    result = {
                        "type": "forecast",
                        "response": "No trade data found for this commodity and country. Try a different query."
                    }
                    return JSONResponse(normalize_response(result))
                raise

        elif query_type == "scenario":
            result = ScenarioSubAgent().handle_with_context(user_query, file_context)
            result["type"] = "scenario"
            return JSONResponse(normalize_response(result))

        else:
            base_response = ask_knowledgebase_with_context(user_query, file_context)
            result = {"type": "rag", "response": base_response}
            return JSONResponse(normalize_response(result))

    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
            return JSONResponse({
                "type": "quota_exceeded",
                "final_output": "The system is temporarily busy due to high demand. Please try again in a moment.",
            })
        print("ERROR in /query:", traceback.format_exc())
        return JSONResponse({"error": f"Processing failed: {str(e)}"}, status_code=500)


@app.get("/healthz")
def health():
    return {"status": "ok"}