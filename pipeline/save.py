import os
import sys
from pathlib import Path

import psycopg2
import requests

sys.path.insert(0, "/opt/airflow/src")

from artifacts import save_model_bundle

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname": os.getenv("POSTGRES_DB", "movielens"),
    "user": os.getenv("POSTGRES_USER", "movielens"),
    "password": os.getenv("POSTGRES_PASSWORD", "movielens"),
}

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODEL_NAME = "two_tower"
K = 10


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def get_best_hit_rate(conn, k: int = K):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(hit_rate) FROM model_metrics WHERE model_name = %s AND k = %s",
            (MODEL_NAME, k),
        )
        row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else None


def save_metrics(conn, metrics: dict, k: int = K):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO model_metrics (model_name, k, hit_rate, ndcg) VALUES (%s, %s, %s, %s)",
            (MODEL_NAME, k, metrics[f"hit_rate@{k}"], metrics[f"ndcg@{k}"]),
        )
    conn.commit()


def notify_api():
    try:
        requests.post("http://api:8001/reload", timeout=5)
        print("Notified API to reload model")
    except Exception as e:
        print(f"Failed to notify API - {e}")


def run_save(model, id_maps: dict, metrics: dict, k: int = K):
    conn = get_conn()
    try:
        best_hit_rate = get_best_hit_rate(conn, k=k)
        new_hit_rate = metrics[f"hit_rate@{k}"]

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if best_hit_rate is None or new_hit_rate > best_hit_rate:
            save_model_bundle(model, id_maps, MODELS_DIR)
            notify_api()
            print(f"New model is better (hit_rate {new_hit_rate} > {best_hit_rate}) - deployed.")
        else:
            print(f"New model is not better (hit_rate {new_hit_rate} <= {best_hit_rate}) - not deployed.")

        save_metrics(conn, metrics, k=k)
    finally:
        conn.close()
