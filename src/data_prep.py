import pandas as pd
import psycopg2
import os
from pathlib import Path

RATINGS_PATH = Path("data/ml-1m/ratings.dat")
POSITIVE_THRESHOLD = 3.0
MIN_POSITIVES_PER_USER = 15
K_TEST = 5


def load_ratings(path: Path = RATINGS_PATH) -> pd.DataFrame:
    """Load MovieLens ratings.dat into a DataFrame """
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )


def keep_positive_ratings(
    ratings_df: pd.DataFrame,
    threshold: float = POSITIVE_THRESHOLD,
) -> pd.DataFrame:
    """Keep only ratings that we treat as positive interactions."""
    return ratings_df[ratings_df["rating"] >= threshold].copy()


def filter_eligible_users(
    positives: pd.DataFrame,
    min_positives: int = MIN_POSITIVES_PER_USER,
) -> pd.DataFrame:
    """Keep users with enough positive interactions for train/test splitting."""
    positives_per_user = positives.groupby("user_id").size()
    eligible_user_ids = positives_per_user[positives_per_user >= min_positives].index
    return positives[positives["user_id"].isin(eligible_user_ids)].copy()


def reindex_ids(filtered_positives: pd.DataFrame):
    """Map raw MovieLens user_id/movie_id values to contiguous embedding indices."""
    user_ids = sorted(filtered_positives["user_id"].unique())
    movie_ids = sorted(filtered_positives["movie_id"].unique())

    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    reindexed = filtered_positives.copy()
    reindexed["user_idx"] = reindexed["user_id"].map(user_id_to_idx)
    reindexed["movie_idx"] = reindexed["movie_id"].map(movie_id_to_idx)

    id_maps = {
        "user_id_to_idx": user_id_to_idx,
        "movie_id_to_idx": movie_id_to_idx,
    }
    return reindexed, id_maps


def leave_k_out_split(
    reindexed_positives: pd.DataFrame,
    k_test: int = K_TEST,
):
    """
    Per user, sort positives by timestamp.
    Last k_test positives become test, all earlier positives become train.
    """
    train_parts = []
    test_parts = []

    for _, user_df in reindexed_positives.groupby("user_id"):
        user_df = user_df.sort_values("timestamp")
        train_parts.append(user_df.iloc[:-k_test])
        test_parts.append(user_df.iloc[-k_test:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, test_df


def prepare_data_from_df(ratings_df: pd.DataFrame):
    """Prepare an already loaded ratings DataFrame into train/test splits + ID maps """
    ratings_df = ratings_df.drop_duplicates(subset=["user_id", "movie_id"]).copy()

    positives = keep_positive_ratings(ratings_df)
    filtered = filter_eligible_users(positives)
    reindexed, id_maps = reindex_ids(filtered)
    train_df, test_df = leave_k_out_split(reindexed)

    print(f"Ratings loaded: {len(ratings_df):,} | positives: {len(positives):,}")
    print(f"Eligible users: {len(id_maps['user_id_to_idx']):,} | movies: {len(id_maps['movie_id_to_idx']):,}")
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    return train_df, test_df, id_maps


def prepare_data(path: Path = RATINGS_PATH):
    """ Load ratings from the local .dat file """
    ratings_df = load_ratings(path)
    return prepare_data_from_df(ratings_df)

def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=5432,
        dbname=os.getenv("POSTGRES_DB", "movielens"),
        user=os.getenv("POSTGRES_USER", "movielens"),
        password=os.getenv("POSTGRES_PASSWORD", "movielens"),
    )


def load_ratings_from_db() -> pd.DataFrame:
    """Load ratings from Postgres into a pandas DataFrame """
    conn = get_conn()

    try:
        query = """
            SELECT
                user_id,
                movie_id,
                rating,
                rated_at AS timestamp
            FROM ratings
        """
        ratings_df = pd.read_sql(query, conn)
        return ratings_df

    finally:
        conn.close()


def prepare_data_from_db():
    """Load from Postgres, then run the shared pipeline """
    ratings_df = load_ratings_from_db()
    return prepare_data_from_df(ratings_df)
