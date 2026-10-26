import torch
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 512


class RatingsPairDataset(Dataset):
    """Positive (user_idx, movie_idx) pairs for two-tower training"""

    def __init__(self, df):
        self.user_idx = torch.as_tensor(df["user_idx"].to_numpy(copy=True), dtype=torch.long)
        self.movie_idx = torch.as_tensor(df["movie_idx"].to_numpy(copy=True), dtype=torch.long)

    def __len__(self):
        return len(self.user_idx)

    def __getitem__(self, idx):
        return self.user_idx[idx], self.movie_idx[idx]


def build_train_loader(train_df, batch_size: int = BATCH_SIZE, generator=None):
    """Only the training split needs a DataLoader"""
    dataset = RatingsPairDataset(train_df)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
