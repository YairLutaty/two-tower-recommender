import numpy as np
import torch

from model import TwoTowerModel, EMBEDDING_DIM as DEFAULT_EMBEDDING_DIM
from losses import in_batch_softmax_loss, TEMPERATURE as DEFAULT_TEMPERATURE
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
LEARNING_RATE = 1e-3
N_EPOCHS = 20


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_movie_log_freq(train_df, num_movies: int, device) -> torch.Tensor:
    """Log movie frequency, used as logQ correction """
    counts = np.ones(num_movies, dtype=np.float64)
    movie_counts = train_df["movie_idx"].value_counts()
    counts[movie_counts.index.to_numpy()] += movie_counts.to_numpy()

    freq = counts / counts.sum()
    return torch.log(torch.tensor(freq, dtype=torch.float32, device=device))


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    movie_log_freq=None,
    temperature: float = DEFAULT_TEMPERATURE,
    save_training_checkpoint: bool = True,
) -> float:
    model.train()
    total_loss = 0.0

    for user_idx, movie_idx in train_loader:
        user_idx = user_idx.to(device)
        movie_idx = movie_idx.to(device)

        optimizer.zero_grad()
        user_vectors, movie_vectors = model(user_idx, movie_idx)
        loss = in_batch_softmax_loss(
            user_vectors=user_vectors,
            movie_vectors=movie_vectors,
            movie_idx=movie_idx,
            movie_log_freq=movie_log_freq,
            temperature=temperature,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def run_training(
    train_loader,
    train_df,
    num_users: int,
    num_movies: int,
    n_epochs: int = N_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    temperature: float = DEFAULT_TEMPERATURE,
):
    device = get_device()
    model = TwoTowerModel(num_users, num_movies, embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    movie_log_freq = compute_movie_log_freq(train_df, num_movies, device)

    for epoch in range(1, n_epochs + 1):
        loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            movie_log_freq=movie_log_freq,
            temperature=temperature,
        )
        print(f"Epoch {epoch}/{n_epochs} | loss {loss:.4f}")

    if save_training_checkpoint:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = ARTIFACTS_DIR / "model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model weights to {model_path.resolve()}")

    return model
