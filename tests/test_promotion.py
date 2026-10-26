import importlib.util
import unittest
from pathlib import Path


def load_save_module():
    path = Path(__file__).resolve().parents[1] / "pipeline" / "save.py"
    spec = importlib.util.spec_from_file_location("pipeline_save", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCursor:
    def __init__(self):
        self.sql = ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, parameters):
        self.sql = sql
        self.parameters = parameters

    def fetchone(self):
        return (0.42,)


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()

    def cursor(self):
        return self.cursor_instance


class PromotionTests(unittest.TestCase):
    def test_promotion_compares_against_best_historical_metric(self):
        module = load_save_module()
        connection = FakeConnection()
        value = module.get_best_hit_rate(connection, k=10)

        self.assertEqual(value, 0.42)
        self.assertIn("MAX(hit_rate)", connection.cursor_instance.sql)
        self.assertEqual(connection.cursor_instance.parameters, ("two_tower", 10))


if __name__ == "__main__":
    unittest.main()
