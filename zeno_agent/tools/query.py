import os
from dotenv import load_dotenv
import psycopg2
from typing import List, Dict, Any

from zeno_agent.embedding_utils import encode_query_to_vector, encode_vector_for_postgres

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def query_embeddings(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        query_vector = encode_query_to_vector(query)
        vector_str = encode_vector_for_postgres(query_vector)
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT content, source
            FROM zeno.rag_embeddings
            ORDER BY embedding_vector <-> %s::vector
            LIMIT %s
            """,
            (vector_str, top_k)
        )
        results = [{"content": r[0], "source": r[1]} for r in cur.fetchall()]
        cur.close()
        conn.close()
        return results
    except Exception as e:
        print(f"[Error] query_embeddings failed: {e}")
        return []