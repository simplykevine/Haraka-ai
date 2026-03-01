import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import traceback
import numpy as np

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extensions import register_adapter, AsIs
from sqlalchemy import create_engine, text
from cachetools import TTLCache

from .embedding_utils import encode_query_to_vector, encode_vector_for_postgres

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set.")

engine = create_engine(DATABASE_URL)

db_pool = None
cache = TTLCache(maxsize=1000, ttl=3600)


def adapt_vector(vec):
    if isinstance(vec, np.ndarray):
        vec = vec.tolist()
    return AsIs("'" + ",".join(str(x) for x in vec) + "'")

register_adapter(np.ndarray, adapt_vector)
register_adapter(list, adapt_vector)


def init_db_pool():
    global db_pool
    db_pool = SimpleConnectionPool(1, 20, DATABASE_URL)


def get_db_connection():
    global db_pool
    if db_pool is None:
        init_db_pool()
    return db_pool.getconn()


def release_db_connection(conn):
    global db_pool
    if db_pool and conn is not None:
        try:
            db_pool.putconn(conn)
        except Exception as e:
            print(f"Error releasing connection: {e}")


def get_country_id_by_name(country_name: str) -> int:
    cache_key = f"country_{country_name.lower()}"
    if cache_key in cache:
        return cache[cache_key]
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM zeno.countries WHERE LOWER(name) = LOWER(%s)",
            (country_name.strip(),)
        )
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Country '{country_name}' not found in zeno.countries.")
        cache[cache_key] = result[0]
        return result[0]
    finally:
        if conn is not None:
            cur.close()
            release_db_connection(conn)


def get_product_id_by_name(product_name: str) -> int:
    cache_key = f"product_{product_name.lower()}"
    if cache_key in cache:
        return cache[cache_key]
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM zeno.products WHERE LOWER(name) = LOWER(%s)",
            (product_name.strip(),)
        )
        result = cur.fetchone()
        if result:
            cache[cache_key] = result[0]
            return result[0]
        cur.execute(
            "SELECT id FROM zeno.products WHERE LOWER(name) LIKE %s",
            (f"%{product_name.strip().lower()}%",)
        )
        result = cur.fetchone()
        if result:
            cache[cache_key] = result[0]
            return result[0]
        raise ValueError(f"Product '{product_name}' not found in zeno.products.")
    finally:
        if conn is not None:
            cur.close()
            release_db_connection(conn)


def get_indicator_id_by_metric(metric: str) -> int:
    cache_key = f"indicator_{metric.lower()}"
    if cache_key in cache:
        return cache[cache_key]
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM zeno.indicators WHERE LOWER(name) LIKE %s LIMIT 1",
            (f"%{metric.strip().lower()}%",)
        )
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Indicator '{metric}' not found in zeno.indicators.")
        cache[cache_key] = result[0]
        return result[0]
    finally:
        if conn is not None:
            cur.close()
            release_db_connection(conn)


def get_trade_data_from_db(
    country_id: int,
    product_id: int,
    indicator_id: int,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> Any:
    import pandas as pd
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = """
            SELECT date, quantity, price, currency, source, metadata
            FROM zeno.trade_data
            WHERE country_id = %s
              AND product_id = %s
              AND indicator_id = %s
        """
        params = [country_id, product_id, indicator_id]
        if start_year:
            query += " AND EXTRACT(YEAR FROM date) >= %s"
            params.append(start_year)
        if end_year:
            query += " AND EXTRACT(YEAR FROM date) <= %s"
            params.append(end_year)
        query += " ORDER BY date ASC"
        cur.execute(query, params)
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["date", "quantity", "price", "currency", "source", "metadata"])
        return df
    except Exception as e:
        print(f"[Error] get_trade_data_from_db failed: {e}")
        traceback.print_exc()
        import pandas as pd
        return pd.DataFrame()
    finally:
        if conn is not None:
            cur.close()
            release_db_connection(conn)


def query_rag_embeddings_semantic(embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        vector_str = encode_vector_for_postgres(embedding)
        cur.execute("""
            SELECT embedding_id, content, source
            FROM zeno.rag_embeddings
            ORDER BY embedding_vector <-> %s::vector
            LIMIT %s
        """, (vector_str, top_k))
        rows = cur.fetchall()
        return [{"embedding_id": r[0], "content": r[1], "source": r[2]} for r in rows]
    except Exception as e:
        print(f"[Error] query_rag_embeddings_semantic failed: {e}")
        traceback.print_exc()
        return []
    finally:
        if conn is not None:
            cur.close()
            release_db_connection(conn)


def query_structured_embeddings_semantic(embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        vector_str = encode_vector_for_postgres(embedding)
        cur.execute("""
            SELECT embedding_id, content, source, document_id
            FROM zeno.structured_embeddings
            ORDER BY embedding_vector <-> %s::vector
            LIMIT %s
        """, (vector_str, top_k))
        rows = cur.fetchall()
        return [{"embedding_id": r[0], "content": r[1], "source": r[2], "document_id": r[3]} for r in rows]
    except Exception as e:
        print(f"[Error] query_structured_embeddings_semantic failed: {e}")
        traceback.print_exc()
        return []
    finally:
        if conn is not None:
            cur.close()
            release_db_connection(conn)