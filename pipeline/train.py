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


_dataset = _load_src_module("_two_tower_dataset", "dataset.py")
_train_core = _load_src_module("_two_tower_train", "train.py")


EMBEDDING_DIM = 128
TEMPERATURE = 0.1
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
N_EPOCHS = 20


def run_training(train_df, num_users: int, num_movies: int):
    """ Wrapper: build the DataLoader, transfer to src/train.py. """
    train_loader = _dataset.build_train_loader(train_df, batch_size=BATCH_SIZE)
    return _train_core.run_training(
        train_loader=train_loader,
        train_df=train_df,
        num_users=num_users,
        num_movies=num_movies,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        embedding_dim=EMBEDDING_DIM,
        temperature=TEMPERATURE,
        save_training_checkpoint=False,
    )
