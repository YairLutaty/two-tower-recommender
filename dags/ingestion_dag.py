import sys
sys.path.insert(0, '/opt/airflow/ingestion')
sys.path.insert(0, '/opt/airflow/pipeline')

from datetime import datetime
from airflow import DAG
from pathlib import Path
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor

import ingest
from prepare import prepare_data
from train import run_training
from evaluate import run_evaluation
from save import run_save


RAW_DIR = '/opt/airflow/data/ml-1m'


def run_pipeline(**context):
    train_df, test_df, id_maps = prepare_data()

    num_users = len(id_maps["user_id_to_idx"])
    num_movies = len(id_maps["movie_id_to_idx"])

    model = run_training(train_df, num_users=num_users, num_movies=num_movies)
    metrics = run_evaluation(model, train_df, test_df, num_movies=num_movies)
    run_save(model, id_maps, metrics)


with DAG(
    dag_id='movielens_ingestion',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    wait_for_users = FileSensor(
        task_id="wait_for_users",
        filepath=f"{RAW_DIR}/users.dat",
        poke_interval=30,
        timeout=86400,
        mode="poke",
    )

    wait_for_movies = FileSensor(
        task_id='wait_for_movies',
        filepath=f'{RAW_DIR}/movies.dat',
        poke_interval=30,
        timeout=86400,
        mode='poke',
    )

    wait_for_ratings = FileSensor(
        task_id='wait_for_ratings',
        filepath=f'{RAW_DIR}/ratings.dat',
        poke_interval=30,
        timeout=86400,
        mode='poke',
    )

    ingest_movies_task = PythonOperator(
        task_id='ingest_movies',
        python_callable=ingest.run_movies,
    )

    ingest_ratings_task = PythonOperator(
        task_id='ingest_ratings',
        python_callable=ingest.run_ratings,
    )

    ingest_users_task = PythonOperator(
        task_id="ingest_users",
        python_callable=ingest.run_users,
    )

    pipeline_task = PythonOperator(
        task_id='run_pipeline',
        python_callable=run_pipeline,
    )

    wait_for_users >> ingest_users_task
    wait_for_movies >> ingest_movies_task
    wait_for_ratings >> ingest_ratings_task
    # Ratings reference users and movies, so their ingestion cannot race the
    # two parent tables even though the file sensors run concurrently.
    [wait_for_ratings, ingest_users_task, ingest_movies_task] >> ingest_ratings_task
    ingest_ratings_task >> pipeline_task
