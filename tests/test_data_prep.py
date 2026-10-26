import unittest

import pandas as pd

from data_prep import prepare_data_from_df


class DataPreparationTests(unittest.TestCase):
    def test_filter_reindex_and_chronological_leave_five_out(self):
        rows = [
            {"user_id": 7, "movie_id": movie, "rating": 4.0, "timestamp": movie}
            for movie in range(1, 17)
        ]
        rows.extend(
            {"user_id": 8, "movie_id": 100 + movie, "rating": 5.0, "timestamp": movie}
            for movie in range(10)
        )
        rows.append({"user_id": 7, "movie_id": 99, "rating": 2.0, "timestamp": 99})

        train, test, maps = prepare_data_from_df(pd.DataFrame(rows))

        self.assertEqual(set(train["movie_id"]), set(range(1, 12)))
        self.assertEqual(set(test["movie_id"]), set(range(12, 17)))
        self.assertEqual(set(train["user_id"]), {7})
        self.assertEqual(maps["user_id_to_idx"], {7: 0})
        self.assertNotIn(99, maps["movie_id_to_idx"])


if __name__ == "__main__":
    unittest.main()
