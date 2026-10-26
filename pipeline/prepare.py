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


_data_prep = _load_src_module("_two_tower_data_prep", "data_prep.py")


def prepare_data():
    return _data_prep.prepare_data_from_db()