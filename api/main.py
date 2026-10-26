import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from threading import RLock

import psycopg2
import torch
from fastapi import FastAPI, HTTPException, Query


sys.path.insert(0, "/app/src")

from artifacts import load_model_bundle


DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "airflow-postgres"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname": os.getenv("POSTGRES_DB", "airflow"),
    "user": os.getenv("POSTGRES_USER", "airflow"),
    "password": os.getenv("POSTGRES_PASSWORD", "airflow"),
}

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
MODEL_PATH = MODELS_DIR / "model.pt"
ID_MAPS_PATH = MODELS_DIR / "id_maps.pkl"

DEVICE = torch.device("cpu")

model = None
id_maps = None
idx_to_movie_id = None
movie_vectors = None
model_lock = RLock()


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def load_model():
    """
    Load: models/model.pt & models/id_maps.pkl
    Then precompute all movie vectors once.
    """
    global model, id_maps, idx_to_movie_id, movie_vectors

    (
        loaded_model,
        loaded_id_maps,
        loaded_idx_to_movie_id,
        loaded_movie_vectors,
    ) = load_model_bundle(MODELS_DIR, DEVICE)

    # Publish the coupled objects together only after the complete bundle
    # validates. Requests take the same lock only long enough to snapshot them.
    with model_lock:
        model = loaded_model
        id_maps = loaded_id_maps
        idx_to_movie_id = loaded_idx_to_movie_id
        movie_vectors = loaded_movie_vectors

    num_users = len(id_maps["user_id_to_idx"])
    num_movies = len(id_maps["movie_id_to_idx"])
    embedding_dim = model.user_embedding.embedding_dim

    print(
        f"Model loaded from {MODEL_PATH} | "
        f"users={num_users:,} | movies={num_movies:,} | dim={embedding_dim}"
    )


def get_seen_movie_ids(user_id: int) -> set[int]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT movie_id FROM ratings WHERE user_id = %s",
                (user_id,),
            )
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def get_movie_titles(movie_ids: list[int]) -> dict[int, str]:
    movie_ids = [int(movie_id) for movie_id in movie_ids]

    if not movie_ids:
        return {}

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT movie_id, title FROM movies WHERE movie_id = ANY(%s)",
                (movie_ids,),
            )
            return {int(row[0]): row[1] for row in cur.fetchall()}
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as e:
        print(f"Model not loaded on startup: {e}")

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    with model_lock:
        is_loaded = model is not None
    return {
        "status": "ok",
        "model_loaded": is_loaded,
        "model_path": str(MODEL_PATH),
        "id_maps_path": str(ID_MAPS_PATH),
    }


@app.post("/reload")
def reload_model():
    try:
        load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "model_reloaded",
    }


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = Query(10, ge=1, le=100)):
    with model_lock:
        active_model = model
        active_id_maps = id_maps
        active_idx_to_movie_id = idx_to_movie_id
        active_movie_vectors = movie_vectors

    if (
        active_model is None
        or active_id_maps is None
        or active_idx_to_movie_id is None
        or active_movie_vectors is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Train/save a model first, then call /reload.",
        )

    user_id_to_idx = active_id_maps["user_id_to_idx"]
    movie_id_to_idx = active_id_maps["movie_id_to_idx"]

    if user_id not in user_id_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} was not in the trained model.",
        )

    user_idx = user_id_to_idx[user_id]

    with torch.no_grad():
        user_tensor = torch.tensor([user_idx], dtype=torch.long, device=DEVICE)
        user_vector = active_model.encode_users(user_tensor)

        scores = (user_vector @ active_movie_vectors.T).squeeze(0)

        seen_movie_ids = get_seen_movie_ids(user_id)

        seen_movie_indices = [
            movie_id_to_idx[movie_id]
            for movie_id in seen_movie_ids
            if movie_id in movie_id_to_idx
        ]

        if seen_movie_indices:
            scores[torch.tensor(seen_movie_indices, dtype=torch.long, device=DEVICE)] = float("-inf")

        top_k_indices = torch.topk(scores, k=min(k, len(scores))).indices.cpu().tolist()

    recommended_movie_ids = [int(active_idx_to_movie_id[idx]) for idx in top_k_indices]
    titles = get_movie_titles(recommended_movie_ids)

    return {
        "user_id": user_id,
        "k": k,
        "recommendations": [
            {
                "movie_id": movie_id,
                "title": titles.get(movie_id, "Missing title"),
            }
            for movie_id in recommended_movie_ids
        ],
    }
