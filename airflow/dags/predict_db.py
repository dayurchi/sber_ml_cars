from datetime import datetime

import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from sqlalchemy import create_engine, Table, MetaData

from ml_cars import predict


INPUT_SESSIONS_PATH = '/opt/airflow/dags/sessions.csv'
INPUT_HITS_PATH = '/opt/airflow/dags/hits.csv'
OUTPUT_PREDICTION_PATH = '/opt/airflow/dags/prediction.csv'


def save_prediction_to_db(
    session_id,
    prediction,
    date_time,
    engine,
    table_name='predictions'
):
    table = Table(table_name, MetaData(), autoload_with=engine)
    with engine.connect() as connection:
        connection.execute(
            table.insert(), {
                'session_id': session_id,
                'prediction': prediction,
                'date_time': date_time
            }
        )


def make_prediction():

    print('Data load started ...')
    sessions = pd.read_csv(INPUT_SESSIONS_PATH, index_col=0)
    hits = pd.read_csv(INPUT_HITS_PATH, index_col=0)
    print('Data load finished ...')

    print('Prediction started ...')
    date_time = datetime.now()
    predictions = predict(sessions, hits)
    print('Prediction finished ...')

    print('Creating a DB connection ...')
    engine = create_engine('sqlite:////app/database.db')

    print('Uploading into DB ...')
    for session_id, prediction in predictions.to_dict().items():
        save_prediction_to_db(session_id, prediction, date_time, engine, 'predictions')
    print('Success!')


dag = DAG(
    'catboost_prediction_db',
    description='Make predictions using a CatBoost model and hits and sessions and save to a SQLite database',
    schedule_interval='@once',
    start_date=datetime(2023, 5, 27)
)

make_prediction_task = PythonOperator(
    task_id='make_prediction',
    python_callable=make_prediction,
    dag=dag
)
