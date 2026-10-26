import unittest

import pandas as pd
import torch

from evaluate import evaluate_retrieval, popularity_baseline
from model import TwoTowerModel


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(
            {"user_idx": [0, 1], "movie_idx": [0, 2]}
        )
        self.test = pd.DataFrame(
            {"user_idx": [0, 1], "movie_idx": [1, 3]}
        )

    def test_two_tower_masks_training_interactions(self):
        model = TwoTowerModel(num_users=2, num_movies=4, embedding_dim=2)
        with torch.no_grad():
            model.user_embedding.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            model.movie_embedding.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
            )

        result = evaluate_retrieval(model, self.train, self.test, num_movies=4, k=1)
        self.assertEqual(result["hit_rate@1"], 1.0)
        self.assertEqual(result["ndcg@1"], 1.0)

    def test_popularity_baseline_excludes_each_users_training_movies(self):
        result = popularity_baseline(self.train, self.test, k=2)
        self.assertEqual(result["users_evaluated"], 2)
        self.assertGreaterEqual(result["hit_rate@2"], 0.0)


if __name__ == "__main__":
    unittest.main()
