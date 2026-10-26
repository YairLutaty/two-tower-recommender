import numpy as np
import pandas as pd
import torch

K = 10


def build_user_positive_lookup(df: pd.DataFrame) -> dict:
    return df.groupby("user_idx")["movie_idx"].apply(set).to_dict()


@torch.no_grad()
def evaluate_retrieval(model, train_df, test_df, num_movies: int, k: int = K, device=None) -> dict:
    """Top-k retrieval evaluation."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    train_seen = build_user_positive_lookup(train_df)
    test_targets = build_user_positive_lookup(test_df)
    users = sorted(test_targets.keys())

    user_idx = torch.tensor(users, device=device)
    movie_idx = torch.arange(num_movies, device=device)

    user_vectors = model.encode_users(user_idx)
    movie_vectors = model.encode_movies(movie_idx)
    scores = user_vectors @ movie_vectors.T

    for row, user in enumerate(users):
        seen = train_seen.get(user, set())
        if seen:
            scores[row, torch.tensor(list(seen), device=device)] = float("-inf")

    top_k = torch.topk(scores, k, dim=1).indices.cpu().numpy()

    hit_scores = []
    ndcg_scores = []

    for row, user in enumerate(users):
        targets = test_targets[user]
        recommended = top_k[row]

        relevance = [1 if movie in targets else 0 for movie in recommended]
        hit_scores.append(1 if any(relevance) else 0)

        dcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(relevance))
        ideal_dcg = sum(1 / np.log2(rank + 2) for rank in range(min(len(targets), k)))
        ndcg_scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

    return {
        f"hit_rate@{k}": round(float(np.mean(hit_scores)), 4),
        f"ndcg@{k}": round(float(np.mean(ndcg_scores)), 4),
        "users_evaluated": len(users),
    }


def popularity_baseline(train_df, test_df, k: int = K) -> dict:
    """ Recommend globally popular train movies, without each user train movies """
    train_seen = build_user_positive_lookup(train_df)
    test_targets = build_user_positive_lookup(test_df)
    ranked_movies = train_df["movie_idx"].value_counts().index.to_numpy()

    hit_scores = []
    ndcg_scores = []

    for user, targets in test_targets.items():
        seen = train_seen.get(user, set())
        recommended = [movie for movie in ranked_movies if movie not in seen][:k]

        relevance = [1 if movie in targets else 0 for movie in recommended]
        hit_scores.append(1 if any(relevance) else 0)

        dcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(relevance))
        ideal_dcg = sum(1 / np.log2(rank + 2) for rank in range(min(len(targets), k)))
        ndcg_scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

    return {
        f"hit_rate@{k}": round(float(np.mean(hit_scores)), 4),
        f"ndcg@{k}": round(float(np.mean(ndcg_scores)), 4),
        "users_evaluated": len(test_targets),
    }


def run_evaluation(model, train_df, test_df, num_movies: int, k: int = K):
    model_results = evaluate_retrieval(model, train_df, test_df, num_movies, k=k)
    baseline_results = popularity_baseline(train_df, test_df, k=k)

    print("\nFinal test results")
    print("------------------")
    for name, value in model_results.items():
        print(f"model       {name}: {value}")
    for name, value in baseline_results.items():
        print(f"popularity  {name}: {value}")

    return model_results, baseline_results
