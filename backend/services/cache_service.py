import sqlite3
import json 
import  numpy as np 
from typing import Optional
from sentence_transformers import SentenceTransformer

CACHE_DB_PATH = "cache.db"
SIMILARITY_THRESHOLD = 0.85

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def create_cache_db():
    with sqlite3.connect(
        CACHE_DB_PATH,
        check_same_thread = False
    ) as conn:
        c = conn.cursor()

        c.execute("""
CREATE TABLE IF NOT EXISTS cache (
                  id INTEGER PRIMARY KEY,
                  question TEXT ,
                  question_vec TEXT,
                  answer TEXT
                  )
                  """)
        conn.commit()

def embed_question(question:str)->list:
    vector = model.encode(
        [question],
        normalize_embeddings=True
    )[0]
    return vector.tolist()

def get_cached_answer(question: str)-> Optional[str]:
    q_vec = embed_question(question)

    with sqlite3.connect(
        CACHE_DB_PATH,
        check_same_thread=False
    ) as  conn:
        c = conn.cursor()
        c.execute(
            
            "SELECT question, question_vec, answer FROM cache"
        )
        rows = c.fetchall()

    best_answer = None
    best_sim = 0.0
    q_np = np.array(q_vec)
    for _, stored_vec_str, stored_answer in rows:
        stored_vec = np.array(
            json.loads(stored_vec_str)
        )

        sim = np.dot(q_np, stored_vec)

        if sim > best_sim:
            best_sim = sim
            best_answer = stored_answer
    if best_sim >= SIMILARITY_THRESHOLD:
        return best_answer
    return None

def add_to_cache(question: str,answer: str):
    q_vec = embed_question(question)

    with sqlite3.connect(
        CACHE_DB_PATH,
        check_same_thread=False
    ) as conn:
        c = conn.cursor()

        c.execute(
            """
            INSERT INTO cache
            (question, question_vec,answer)
            VALUES (?,?,?)
            """,
            (
                question,
                json.dumps(q_vec),
                answer
            )
        )

        conn.commit()