"""Validated persistence for the model and its coupled ID mappings."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import torch

from model import TwoTowerModel


MODEL_FILENAME = "model.pt"
ID_MAPS_FILENAME = "id_maps.pkl"


def _validate_id_maps(id_maps: dict) -> tuple[dict, dict]:
    required = {"user_id_to_idx", "movie_id_to_idx"}
    if set(id_maps) != required:
        raise ValueError(f"ID maps must contain exactly {sorted(required)}")

    user_map = id_maps["user_id_to_idx"]
    movie_map = id_maps["movie_id_to_idx"]
    for name, mapping in (("user", user_map), ("movie", movie_map)):
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError(f"{name} indices must be contiguous from zero")
    return user_map, movie_map


def save_model_bundle(model, id_maps: dict, output_dir: Path | str) -> None:
    """Write weights and mappings with replace-on-success semantics."""
    _validate_id_maps(id_maps)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / MODEL_FILENAME
    maps_path = output_dir / ID_MAPS_FILENAME
    model_tmp = output_dir / f".{MODEL_FILENAME}.tmp"
    maps_tmp = output_dir / f".{ID_MAPS_FILENAME}.tmp"

    try:
        torch.save(model.state_dict(), model_tmp)
        with maps_tmp.open("wb") as handle:
            pickle.dump(id_maps, handle)
        os.replace(model_tmp, model_path)
        os.replace(maps_tmp, maps_path)
    finally:
        model_tmp.unlink(missing_ok=True)
        maps_tmp.unlink(missing_ok=True)


def load_model_bundle(
    model_dir: Path | str,
    device: torch.device | str = "cpu",
) -> tuple[TwoTowerModel, dict, dict[int, int], torch.Tensor]:
    """Load and validate a model bundle before callers swap live state."""
    model_dir = Path(model_dir)
    model_path = model_dir / MODEL_FILENAME
    maps_path = model_dir / ID_MAPS_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not maps_path.exists():
        raise FileNotFoundError(f"Missing ID maps file: {maps_path}")

    # Pickle is retained for compatibility with the original project. Only
    # load bundles produced locally by this repository.
    with maps_path.open("rb") as handle:
        id_maps = pickle.load(handle)
    user_map, movie_map = _validate_id_maps(id_maps)

    resolved_device = torch.device(device)
    state_dict = torch.load(model_path, map_location=resolved_device, weights_only=True)
    required_weights = {"user_embedding.weight", "movie_embedding.weight"}
    if set(state_dict) != required_weights:
        raise ValueError("Unexpected model state keys")

    user_weight = state_dict["user_embedding.weight"]
    movie_weight = state_dict["movie_embedding.weight"]
    if user_weight.shape[0] != len(user_map) or movie_weight.shape[0] != len(movie_map):
        raise ValueError("Model dimensions do not match the saved ID mappings")
    if user_weight.shape[1] != movie_weight.shape[1]:
        raise ValueError("User and movie embedding dimensions differ")

    loaded_model = TwoTowerModel(
        num_users=len(user_map),
        num_movies=len(movie_map),
        embedding_dim=user_weight.shape[1],
    ).to(resolved_device)
    loaded_model.load_state_dict(state_dict)
    loaded_model.eval()

    with torch.no_grad():
        movie_indices = torch.arange(len(movie_map), device=resolved_device)
        movie_vectors = loaded_model.encode_movies(movie_indices)
    idx_to_movie_id = {index: movie_id for movie_id, index in movie_map.items()}
    return loaded_model, id_maps, idx_to_movie_id, movie_vectors
