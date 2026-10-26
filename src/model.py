import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 128


class TwoTowerModel(nn.Module):
    """Two-tower model - one embedding table for users, one for movies """

    def __init__(self, num_users: int, num_movies: int, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, mean=0.0, std=0.01)

    def encode_users(self, user_idx: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_embedding(user_idx), p=2, dim=-1)

    def encode_movies(self, movie_idx: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.movie_embedding(movie_idx), p=2, dim=-1)

    def forward(self, user_idx: torch.Tensor, movie_idx: torch.Tensor):
        return self.encode_users(user_idx), self.encode_movies(movie_idx)
