import os
import shutil
import psycopg2
import pandas as pd
from io import StringIO
from pathlib import Path

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname": os.getenv("POSTGRES_DB", "movielens"),
    "user": os.getenv("POSTGRES_USER", "movielens"),
    "password": os.getenv("POSTGRES_PASSWORD", "movielens"),
}

RAW_DIR = Path(os.getenv("RAW_DIR", "data/ml-1m"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "data/ml-1m/processed"))

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def copy_df_to_table(cursor, df, table: str, chunksize: int = 100_000):
    staging = f"staging_{table}"
    cursor.execute(f"CREATE TEMP TABLE {staging} (LIKE {table} INCLUDING ALL) ON COMMIT DROP")

    for i in range(0, len(df), chunksize):
        chunk = df.iloc[i: i + chunksize]
        buffer = StringIO()
        chunk.to_csv(buffer, index=False, header=False)
        buffer.seek(0)
        cursor.copy_expert(f"COPY {staging} FROM STDIN WITH CSV", buffer)

    cursor.execute(f"""
        INSERT INTO {table}
        SELECT * FROM {staging}
        ON CONFLICT DO NOTHING
    """)

def ingest_users(conn, users_path: Path):
    df = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["user_id", "gender", "age", "occupation", "zip_code"]
    )

    users = df[["user_id", "gender", "age", "occupation", "zip_code"]]

    with conn.cursor() as cur:
        copy_df_to_table(cur, users, "users")

    conn.commit()

    print(f"Inserted {len(users):,} users")

def ingest_movies(conn, movies_path: Path):
    df = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["movie_id", "title", "genres"]
    )

    movies = df[["movie_id", "title"]]

    genres = (
        df[["movie_id", "genres"]]
        .assign(genre=df["genres"].str.split("|"))
        .explode("genre")
        [lambda x: x["genre"] != "(no genres listed)"]
        [["movie_id", "genre"]]
    )

    with conn.cursor() as cur:
        copy_df_to_table(cur, movies, "movies")
        copy_df_to_table(cur, genres, "genres")

    conn.commit()

    print(f"Inserted {len(movies)} movies, {len(genres)} genre rows")

def ingest_ratings(conn, ratings_path: Path):
    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    df["rated_at"] = pd.to_datetime(df["timestamp"], unit="s")

    ratings = df[["user_id", "movie_id", "rating", "rated_at"]]

    with conn.cursor() as cur:
        copy_df_to_table(cur, ratings, "ratings")

    conn.commit()

    print(f"Inserted {len(ratings):,} ratings")

def run_users():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    try:
        users_file = RAW_DIR / "users.dat"
        if users_file.exists():
            print("Ingesting users file...")
            ingest_users(conn, users_file)
            shutil.move(str(users_file), PROCESSED_DIR / users_file.name)
    finally:
        conn.close()

def run_movies():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    try:
        movies_file = RAW_DIR / "movies.dat"
        if movies_file.exists():
            print("Ingesting movies file...")
            ingest_movies(conn, movies_file)
            shutil.move(
                str(movies_file),
                PROCESSED_DIR / movies_file.name
            )
    finally:
        conn.close()

def run_ratings():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    try:
        ratings_file = RAW_DIR / "ratings.dat"
        if ratings_file.exists():
            print("Ingesting ratings file...")
            ingest_ratings(conn, ratings_file)
            shutil.move(
                str(ratings_file),
                PROCESSED_DIR / ratings_file.name
            )
    finally:
        conn.close()

def run():
    run_users()
    run_movies()
    run_ratings()

if __name__ == "__main__":
    run()
