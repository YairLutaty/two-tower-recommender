# MovieLens Two-Tower Recommender

An end-to-end movie recommendation project built on the
[MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/). It trains
a two-tower retrieval model with PyTorch, evaluates it against a popularity
baseline, and serves personalized recommendations through FastAPI.

The repository supports two workflows:

- **Local experimentation** for preparing the data, training the model, and
  printing evaluation metrics.
- **Orchestrated deployment** with Airflow, PostgreSQL, Docker Compose, and a
  FastAPI inference service.

I implemented the data preparation and chronological evaluation protocol, the
PyTorch two-tower model and frequency-corrected in-batch loss, the popularity
baseline, the Airflow/PostgreSQL promotion workflow, and the FastAPI serving
path. The earlier KNN and factorization-machine exercises are intentionally not
combined here because their evaluation protocols are not directly comparable.

## Verified archived result

An existing model bundle was reloaded and evaluated against its matching ID
mappings and MovieLens input before this standalone repository was prepared:

| System | Hit Rate@10 | NDCG@10 |
|---|---:|---:|
| Global popularity | 0.1672 | 0.0341 |
| Two-tower | **0.3275** | **0.0718** |

The protocol evaluated 5,988 eligible users, each with their five most recent
positive interactions held out. The exact protocol, input hashes, and result
are recorded in [`results/archived_offline_evaluation.json`](results/archived_offline_evaluation.json).
The retained weights are not redistributed, and that historical training run
did not record a seed; fresh training may therefore produce different values.

## How it works

```text
MovieLens .dat files
        |
        v
Airflow file sensors -> PostgreSQL ingestion
        |
        v
Positive interactions + chronological leave-5-out split
        |
        v
PyTorch two-tower training
        |
        v
Hit Rate@10 and NDCG@10 evaluation
        |
        v
Model promotion -> FastAPI reload -> recommendations
```

The model learns separate, L2-normalized embedding tables for users and
movies. Training uses in-batch softmax negatives with a movie-frequency
correction. At retrieval time, recommendations are ranked by user/movie
embedding similarity and movies already rated by the user are excluded.

### Data preparation

- Ratings of `3.0` or higher are treated as positive interactions.
- Users must have at least 15 positive interactions.
- IDs are mapped to contiguous embedding indices.
- Each user's five most recent positive interactions form the test set.
- All earlier positive interactions form the training set.

### Evaluation

The learned model is compared with a global popularity baseline using:

- **Hit Rate@10**: the fraction of users with at least one held-out movie in
  their top-10 recommendations.
- **NDCG@10**: a rank-sensitive measure of how highly held-out movies appear.

Training hyperparameters are command-line options in `run.py` for local runs and
`pipeline/train.py` for Airflow runs. The defaults are 20 epochs, 128
embedding dimensions, a batch size of 512, a learning rate of `1e-3`, and a
softmax temperature of `0.1`.

## Project structure

```text
.
├── api/
│   ├── Dockerfile
│   └── main.py              # FastAPI inference and model reload endpoints
├── dags/
│   └── ingestion_dag.py     # Airflow ingestion and training DAG
├── db/
│   └── init.sql             # MovieLens and model-metrics tables
├── ingestion/
│   └── ingest.py            # Idempotent .dat-to-PostgreSQL ingestion
├── pipeline/
│   ├── prepare.py           # Airflow wrapper for data preparation
│   ├── train.py             # Airflow wrapper for model training
│   ├── evaluate.py          # Airflow wrapper for evaluation
│   └── save.py              # Metrics, promotion, and API notification
├── src/
│   ├── artifacts.py         # Coupled model/ID-map persistence and validation
│   ├── data_prep.py
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   └── evaluate.py
├── data/ml-1m/              # Raw data drop location
├── models/                  # Model and ID maps consumed by the API
├── results/                 # Small, provenance-bearing result records
├── tests/                   # Offline unit tests
├── docker-compose.yml
├── requirements.txt
└── run.py                   # Local training entry point
```

## Local training

### Prerequisites

- Python 3.12 recommended
- `pip`
- The MovieLens 1M `ratings.dat` file

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download and extract MovieLens 1M, then make sure the ratings file is at:

```text
data/ml-1m/ratings.dat
```

Run training and evaluation from the repository root:

```bash
PYTHONPATH=src python run.py
```

The command prints epoch losses and both evaluations, then writes the coupled
API-ready bundle to `models/model.pt` and `models/id_maps.pkl`. Use only bundles
created by this repository because the mapping file uses Python pickle.

Useful options include `--epochs`, `--seed`, `--ratings`, and `--output-dir`.
To train from PostgreSQL instead, pass `--use-db` and provide
`POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, and
`POSTGRES_PASSWORD`.

Run the lightweight offline tests without downloading MovieLens or starting
Docker:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Dockerized Airflow and API workflow

### Prerequisites

- Docker with Docker Compose
- The three MovieLens 1M files:
  `users.dat`, `movies.dat`, and `ratings.dat`

Copy the local placeholder configuration and replace both placeholder secrets:

```bash
cp .env.example .env
```

Place the dataset files in the raw-data directory:

```text
data/ml-1m/users.dat
data/ml-1m/movies.dat
data/ml-1m/ratings.dat
```

Start the services:

```bash
docker compose up --build airflow-webserver airflow-scheduler api
```

Then open Airflow at [http://localhost:8080](http://localhost:8080), sign in
with username `admin` and password `admin`, and trigger the
`movielens_ingestion` DAG.

The DAG:

1. Waits for all three dataset files.
2. Loads users, movies, genres, and ratings into PostgreSQL.
3. Moves ingested files to `data/ml-1m/processed/`.
4. Prepares the chronological train/test split.
5. Trains and evaluates the two-tower model.
6. Records the evaluation metrics in `model_metrics`.
7. Promotes the new artifacts when their Hit Rate@10 improves on the most
   successful previously recorded run.
8. Calls the API reload endpoint after a successful promotion.

Promoted artifacts are stored in the host-mounted `models/` directory:

```text
models/model.pt
models/id_maps.pkl
```

The ingestion code creates `processed/` when needed. Because it moves the
source files there after successful database writes, copy or extract
fresh files back into `data/ml-1m/` before triggering another ingestion run.

## Recommendation API

The API is available at [http://localhost:8001](http://localhost:8001).
Interactive OpenAPI documentation is available at
[http://localhost:8001/docs](http://localhost:8001/docs).

Check service and model status:

```bash
curl http://localhost:8001/health
```

Request recommendations for a known MovieLens user:

```bash
curl "http://localhost:8001/recommend/1?k=10"
```

Example response:

```json
{
  "user_id": 1,
  "k": 10,
  "recommendations": [
    {
      "movie_id": 1193,
      "title": "One Flew Over the Cuckoo's Nest (1975)"
    }
  ]
}
```

Reload artifacts without restarting the service:

```bash
curl -X POST http://localhost:8001/reload
```

### API behavior

- `/health` responds even if no model is loaded; inspect `model_loaded`.
- `/recommend/{user_id}` returns `404` when the user was not present during
  training.
- Recommendation requests return `503` until both model artifacts have been
  loaded.
- The API queries PostgreSQL to filter rated movies and resolve movie titles.
- The project currently supports known users only; it has no cold-start
  strategy for unseen users.

### Offline evaluation versus serving

Offline evaluation masks only each user's **training** positives, because the
five held-out positives must remain eligible targets. The serving API instead
queries the complete ratings table and removes every movie already rated by the
user. This difference is intentional: evaluation asks whether the model can
recover future held-out interactions, while serving should not recommend items
the user has already consumed. Consequently, a served list is not expected to
match an offline top-10 list for the same historical user.

## Database schema

The PostgreSQL schema contains:

- `users`: MovieLens user attributes
- `movies`: movie IDs and titles
- `genres`: one row per movie/genre pair
- `ratings`: user/movie ratings and timestamps
- `model_metrics`: Hit Rate and NDCG history for trained models

Ingestion uses staging tables and `ON CONFLICT DO NOTHING`, making repeated
loads of the same records safe.

## Current limitations

- User and movie towers are embedding lookups only; metadata such as genre,
  age, and occupation is ingested but not used by the model.
- Retrieval scores all eligible movies in memory, which is appropriate for
  MovieLens 1M but not a large production catalog.
- New local runs seed Python, NumPy, PyTorch, and DataLoader shuffling, but exact
  determinism can still depend on hardware and PyTorch kernels. The archived
  result predates this seed control.
- In-batch negatives can treat another row's duplicate positive movie as a
  negative; the current pipeline does not deduplicate movies within a batch.
- Model promotion uses Hit Rate@10 as its sole acceptance criterion.
- API inference runs on CPU and does not handle unseen users.
- The Compose topology and configuration are statically validated here, but a
  complete Docker/Airflow training run is not part of the lightweight public
  verification.
