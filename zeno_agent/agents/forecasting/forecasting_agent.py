import re
import json
import numpy as np
from .config import client, GENERATION_MODEL, ALIAS_MAP, SUPPORTED_COMMODITIES, SUPPORTED_COUNTRIES
from .data_utils import convert_to_usd, prepare_dual_data, get_enhanced_rag_context
from .model_utils import run_model
from zeno_agent.db_utils import get_country_id_by_name, get_product_id_by_name
from zeno_agent.economist_fallback import economist_web_answer


class ForecastingAgent:

    def parse_timeframe(self, timeframe: str) -> int:
        match = re.match(r"next (\d+) (year|years|month|months)", timeframe.lower())
        if not match:
            return 3
        num, unit = int(match.group(1)), match.group(2)
        return num if "month" in unit else num * 12

    def forecast_dual_metrics(self, df, periods: int):
        unit_df = df[["ds", "unit_price"]].rename(columns={"unit_price": "y"})
        unit_forecast, unit_ints, unit_model = run_model(unit_df, periods, "unit_price")

        rev_df = df[["ds", "price"]].rename(columns={"price": "y"})
        rev_forecast, rev_ints, rev_model = run_model(rev_df, periods, "revenue")

        vol_df = df[["ds", "quantity_kg"]].rename(columns={"quantity_kg": "y"})
        vol_forecast, _, _ = run_model(vol_df, periods, "volume")

        return {
            "unit_price": {"forecast": float(np.mean(unit_forecast)), "intervals": unit_ints, "model": unit_model},
            "total_revenue": {"forecast": float(np.mean(rev_forecast)), "intervals": rev_ints, "model": rev_model},
            "volume_kg": float(np.mean(vol_forecast)),
        }

    def run(self, inputs):
        query = inputs.get("query", "")
        file_context = inputs.get("file_context", "")
        if not query:
            return {"error": "No query provided."}

        q = query.lower()
        commodity = next((k for k in ALIAS_MAP if k in q), None)
        country = next((c for c in SUPPORTED_COUNTRIES if c in q), None)
        timeframe = "next 3 months" if "month" in q else "next 1 year"

        if not commodity or not country:
            # Still try to answer from web even without matching commodity/country
            print("[Forecast] No commodity/country match — using economist web fallback")
            analysis = economist_web_answer(
                query=query,
                agent_type="forecast",
                commodity=commodity or "",
                country=country or ""
            )
            return {
                "type": "forecast",
                "query": query,
                "forecast_display": "See interpretation below.",
                "interpretation": analysis,
                "confidence_level": "Medium",
                "data_points_used": 0,
                "source": "web_fallback"
            }

        commodity = ALIAS_MAP[commodity]

        try:
            country_id = get_country_id_by_name(country.title())
            product_id = get_product_id_by_name(commodity)
        except Exception as e:
            print(f"[Forecast] DB lookup failed — using economist web fallback: {e}")
            analysis = economist_web_answer(query, "forecast", commodity, country)
            return {
                "type": "forecast",
                "query": query,
                "forecast_display": "See interpretation below.",
                "interpretation": analysis,
                "confidence_level": "Medium",
                "data_points_used": 0,
                "source": "web_fallback"
            }

        try:
            df, currency, vol_unit, symbol = prepare_dual_data(country_id, product_id)
        except ValueError as e:
            # No trade data in DB — use web fallback
            print(f"[Forecast] No trade data in DB — using economist web fallback: {e}")
            analysis = economist_web_answer(query, "forecast", commodity, country)
            return {
                "type": "forecast",
                "query": query,
                "forecast_display": "See interpretation below.",
                "interpretation": analysis,
                "confidence_level": "Medium",
                "data_points_used": 0,
                "source": "web_fallback"
            }
        except Exception as e:
            print(f"[Forecast] Data retrieval failed — using economist web fallback: {e}")
            analysis = economist_web_answer(query, "forecast", commodity, country)
            return {
                "type": "forecast",
                "query": query,
                "forecast_display": "See interpretation below.",
                "interpretation": analysis,
                "confidence_level": "Medium",
                "data_points_used": 0,
                "source": "web_fallback"
            }

        rag_context = get_enhanced_rag_context(commodity, country, "price")
        periods = self.parse_timeframe(timeframe)

        try:
            dual_forecast = self.forecast_dual_metrics(df, periods)
        except Exception as e:
            print(f"[Forecast] Model failed — using economist web fallback: {e}")
            analysis = economist_web_answer(query, "forecast", commodity, country)
            return {
                "type": "forecast",
                "query": query,
                "forecast_display": "See interpretation below.",
                "interpretation": analysis,
                "confidence_level": "Medium",
                "data_points_used": len(df),
                "source": "web_fallback"
            }

        display_text = (
            f"Unit Price: {dual_forecast['unit_price']['forecast']:.2f} {currency}/kg | "
            f"Revenue: {dual_forecast['total_revenue']['forecast']:.0f} {currency} | "
            f"Volume: {dual_forecast['volume_kg']:.0f} {vol_unit}"
        )

        prompt = f"""Interpret this forecast professionally for economists.
No markdown, no bullets, just structured paragraphs.

Commodity: {commodity}
Country: {country}
Unit Price: {dual_forecast['unit_price']['forecast']:.2f} {currency}/kg
Total Revenue: {dual_forecast['total_revenue']['forecast']:.0f} {currency}
Volume: {dual_forecast['volume_kg']:.0f} {vol_unit}
Context: {rag_context}
{"Additional document context: " + file_context if file_context else ""}""".strip()

        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=[{"parts": [{"text": prompt}]}],
                config={"max_output_tokens": 1024, "temperature": 0.3}
            )
            if response.candidates and response.candidates[0].content.parts:
                raw_text = response.candidates[0].content.parts[0].text
                interpretation = re.sub(r"[\*\#\-\_\<\>\/]+", "", raw_text).strip()
            else:
                interpretation = economist_web_answer(query, "forecast", commodity, country)
        except Exception as e:
            print(f"[Forecast] LLM interpretation failed — using economist web fallback: {e}")
            interpretation = economist_web_answer(query, "forecast", commodity, country)

        return {
            "type": "forecast",
            "query": query,
            "forecast_display": display_text,
            "dual_forecast": dual_forecast,
            "interpretation": interpretation,
            "confidence_level": "High" if len(df) >= 24 else "Medium",
            "data_points_used": len(df),
        }