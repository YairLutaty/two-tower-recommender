import importlib.util
from pathlib import Path

SRC_DIR = Path("/opt/airflow/src")

import sys
sys.path.append(str(SRC_DIR))


def _load_src_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_evaluate_core = _load_src_module("_two_tower_evaluate", "evaluate.py")

K = 10


def run_evaluation(model, train_df, test_df, num_movies: int):
    model_results, _baseline_results = _evaluate_core.run_evaluation(
        model=model, train_df=train_df, test_df=test_df, num_movies=num_movies, k=K,
    )
    return model_results