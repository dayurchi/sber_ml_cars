from datetime import datetime

import json
import os
import re
import pandas as pd
from pandas import json_normalize
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from sqlalchemy import create_engine, Table, MetaData

from ml_cars import predict


INPUT_JSON_PATH = '/opt/airflow/dags/new_jsons'
OUTPUT_PREDICTION_PATH = '/opt/airflow/dags/prediction_new.json'

def save_prediction_to_db(
        session_id: int,
        prediction: int,
        date_time: datetime,
        engine: create_engine,
        table_name: str = 'predictions'
) -> None:
    table = Table(table_name, MetaData(), autoload_with=engine)
    with engine.connect() as connection:
        connection.execute(
            table.insert(), {
                'session_id': session_id,
                'prediction': prediction,
                'date_time': date_time
            }
        )


def make_prediction() -> None:

    file_list = os.listdir(INPUT_JSON_PATH)

    for file_name in file_list:
        # Путь к файлу
        file_path = os.path.join(INPUT_JSON_PATH, file_name)

        # Извлечение заголовка из названия файла
        match = re.search(r'ga_hits_new_(\d{4}-\d{2}-\d{2})\.json', file_name)
        if match:
            record_path = match.group(1)

            # Чтение JSON
            with open(file_path) as json_file:
                hits_data = json.load(json_file)

            hits = json_normalize(hits_data, record_path=record_path)

        match = re.search(r'ga_sessions_new_(\d{4}-\d{2}-\d{2})\.json', file_name)
        if match:
            record_path = match.group(1)

            with open(file_path) as json_file:
                sessions_data = json.load(json_file)

            sessions = json_normalize(sessions_data, record_path=record_path)

            if not sessions.empty and not hits.empty:
                date_time = datetime.now()
                predictions = predict(sessions, hits)
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
