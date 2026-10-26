import torch
import torch.nn.functional as F

TEMPERATURE = 0.1


def in_batch_softmax_loss(
    user_vectors: torch.Tensor,
    movie_vectors: torch.Tensor,
    movie_idx: torch.Tensor | None = None,
    movie_log_freq: torch.Tensor | None = None,
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    """
    Cross-entropy over in batch movies.
    Row i's correct movie is column i.
    """
    logits = user_vectors @ movie_vectors.T / temperature

    if movie_idx is not None and movie_log_freq is not None:
        logits = logits - movie_log_freq[movie_idx].unsqueeze(0)

    labels = torch.arange(user_vectors.size(0), device=user_vectors.device)
    return F.cross_entropy(logits, labels)
