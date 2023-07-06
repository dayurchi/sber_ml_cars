from datetime import datetime

import os
import json
import pandas as pd
import re
from pandas import json_normalize
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from ml_cars import predict


INPUT_JSON_PATH = '/opt/airflow/dags/new_jsons'
OUTPUT_PREDICTION_PATH = '/opt/airflow/dags/results_jsons'


def make_prediction() -> None:
    def record_json(name: str) -> None:
        new_filename = "prediction_new_" + record_path

        # Обработка разделителей пути с использованием os.path.join()
        output_path = os.path.join(OUTPUT_PREDICTION_PATH, new_filename + ".json")

        # Запись данных в новый файл
        with open(output_path, 'w') as file:
            file.write(name)

    # Получение списка файлов в папке
    file_list = os.listdir(INPUT_JSON_PATH)

    for file_name in file_list:
        # Путь к файлу
        file_path = os.path.join(INPUT_JSON_PATH, file_name)

        #Извлечение заголовка из названия файла
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
                prediction = predict(sessions, hits)
                prediction_json = prediction.to_json(orient='index')
                record_json(prediction_json)
            else:
                prediction_json_empty = "{}"  # Пустой JSON
                record_json(prediction_json_empty)


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
