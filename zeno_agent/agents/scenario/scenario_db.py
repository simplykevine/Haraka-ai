import pandas as pd
from zeno_agent.db_utils import (
    get_country_id_by_name,
    get_product_id_by_name,
    get_indicator_id_by_metric,
    get_trade_data_from_db,
    get_db_connection,
    release_db_connection,
)


def get_macro_stats_from_db(country_id, indicator_id, start_year=2015):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT EXTRACT(YEAR FROM date)::int AS year, AVG(quantity) AS value
            FROM zeno.trade_data
            WHERE country_id = %s
              AND indicator_id = %s
              AND date >= make_date(%s, 1, 1)
            GROUP BY EXTRACT(YEAR FROM date)
            ORDER BY year ASC
        """, (country_id, indicator_id, start_year))
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=["year", "value"])
    except Exception as e:
        print(f"[Scenario] get_macro_stats_from_db error: {e}")
        return pd.DataFrame()
    finally:
        if conn is not None:
            cur.close()
            release_db_connection(conn)


def build_structured_context(commodity, country):
    context_parts = []

    try:
        country_id = get_country_id_by_name(country.title())
    except Exception as e:
        print(f"[Scenario] Country lookup error: {e}")
        country_id = None

    try:
        product_id = get_product_id_by_name(commodity)
    except Exception as e:
        print(f"[Scenario] Product lookup error: {e}")
        product_id = None

    if country_id and product_id:
        try:
            indicator_id = get_indicator_id_by_metric("price")
            df_price = get_trade_data_from_db(country_id, product_id, indicator_id)
            if not df_price.empty:
                avg_price = df_price["price"].mean()
                currency = df_price["currency"].iloc[0] if "currency" in df_price.columns else "KES"
                context_parts.append(f"Average historical price: {currency} {avg_price:,.2f} per unit.")
        except Exception:
            pass

        try:
            indicator_id = get_indicator_id_by_metric("quantity")
            df_qty = get_trade_data_from_db(country_id, product_id, indicator_id)
            if not df_qty.empty:
                total_qty = df_qty["quantity"].sum()
                context_parts.append(f"Historical export volume: {total_qty:,.0f} units.")
        except Exception:
            pass

    macro_indicators = ["GDP", "CPI", "Inflation", "Trade Balance"]
    for macro in macro_indicators:
        try:
            indicator_id = get_indicator_id_by_metric(macro)
            df_macro = get_macro_stats_from_db(country_id, indicator_id, start_year=2015)
            if not df_macro.empty:
                recent_year = df_macro["year"].max()
                recent_value = df_macro[df_macro["year"] == recent_year]["value"].iloc[0]
                context_parts.append(f"{macro} ({recent_year}): {recent_value:,.2f}")
        except Exception:
            continue

    return " | ".join(context_parts) if context_parts else "No structured economic data available."