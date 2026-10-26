"""Train and evaluate the MovieLens two-tower model locally."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from artifacts import save_model_bundle
from data_prep import load_ratings, prepare_data_from_db, prepare_data_from_df
from dataset import build_train_loader
from evaluate import run_evaluation
from train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=Path("data/ml-1m/ratings.dat"))
    parser.add_argument("--use-db", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.use_db:
        train_df, test_df, id_maps = prepare_data_from_db()
    else:
        train_df, test_df, id_maps = prepare_data_from_df(load_ratings(args.ratings))

    num_users = len(id_maps["user_id_to_idx"])
    num_movies = len(id_maps["movie_id_to_idx"])
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = build_train_loader(
        train_df,
        batch_size=args.batch_size,
        generator=generator,
    )
    model = run_training(
        train_loader=train_loader,
        train_df=train_df,
        num_users=num_users,
        num_movies=num_movies,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        temperature=args.temperature,
        save_training_checkpoint=False,
    )
    model_results, baseline_results = run_evaluation(
        model=model,
        train_df=train_df,
        test_df=test_df,
        num_movies=num_movies,
        k=args.k,
    )
    save_model_bundle(model, id_maps, args.output_dir)
    print(f"Saved API-ready model bundle to {args.output_dir.resolve()}")
    print({"two_tower": model_results, "popularity": baseline_results})


if __name__ == "__main__":
    main()
