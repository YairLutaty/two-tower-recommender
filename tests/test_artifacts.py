import tempfile
import unittest
from pathlib import Path

import torch

from artifacts import load_model_bundle, save_model_bundle
from model import TwoTowerModel


class ArtifactTests(unittest.TestCase):
    def test_model_and_mappings_round_trip_as_one_bundle(self):
        model = TwoTowerModel(num_users=2, num_movies=3, embedding_dim=4)
        maps = {
            "user_id_to_idx": {10: 0, 11: 1},
            "movie_id_to_idx": {20: 0, 21: 1, 22: 2},
        }
        with tempfile.TemporaryDirectory() as directory:
            save_model_bundle(model, maps, directory)
            loaded, loaded_maps, reverse_movies, movie_vectors = load_model_bundle(directory)

        self.assertEqual(loaded_maps, maps)
        self.assertEqual(reverse_movies, {0: 20, 1: 21, 2: 22})
        self.assertEqual(tuple(movie_vectors.shape), (3, 4))
        self.assertFalse(loaded.training)
        self.assertTrue(torch.allclose(movie_vectors.norm(dim=1), torch.ones(3)))

    def test_non_contiguous_mapping_is_rejected(self):
        model = TwoTowerModel(num_users=1, num_movies=1, embedding_dim=2)
        bad_maps = {
            "user_id_to_idx": {10: 2},
            "movie_id_to_idx": {20: 0},
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "contiguous"):
                save_model_bundle(model, bad_maps, Path(directory))


if __name__ == "__main__":
    unittest.main()
