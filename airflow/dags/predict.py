from datetime import datetime

import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from ml_cars import predict


INPUT_SESSIONS_PATH = '/opt/airflow/dags/sessions.csv'
INPUT_HITS_PATH = '/opt/airflow/dags/hits.csv'
OUTPUT_PREDICTION_PATH = '/opt/airflow/dags/prediction.csv'


def make_prediction():

    print('Data load started ...')
    sessions = pd.read_csv(INPUT_SESSIONS_PATH, index_col=0)
    hits = pd.read_csv(INPUT_HITS_PATH, index_col=0)
    print('Data load finished ...')

    print('Prediction started ...')
    prediction = predict(sessions, hits)
    prediction.to_csv(OUTPUT_PREDICTION_PATH)
    print('Prediction finished ...')


dag = DAG(
    'catboost_prediction',
    description='Make predictions using a CatBoost model and hits and sessions',
    schedule_interval='@once',
    start_date=datetime(2023, 5, 27)
)

make_prediction_task = PythonOperator(
    task_id='make_prediction',
    python_callable=make_prediction,
    dag=dag
)
