from datetime import datetime

import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from sqlalchemy import create_engine, Table, MetaData

from ml_cars import load_model, process_hits, process_sessions, process_features


COLUMNS_TO_DROP = ['client_id']

INPUT_SESSIONS_PATH = '/opt/airflow/dags/sessions.csv'
TMP_SESSIONS_PATH = '/opt/airflow/dags/sessions_tmp.csv'
INPUT_HITS_PATH = '/opt/airflow/dags/hits.csv'
TMP_UTM_SOURCE = '/opt/airflow/dags/utm_source_tmp.csv'
TMP_CARS = '/opt/airflow/dags/cars_tmp.csv'
TMP_MODELS = '/opt/airflow/dags/models_tmp.csv'
TMP_IS_ACTION = '/opt/airflow/dags/is_action_tmp.csv'
OUTPUT_PREDICTION_PATH = '/opt/airflow/dags/prediction.csv'


def save_prediction_to_db() -> None:
    y_pred = pd.read_csv(OUTPUT_PREDICTION_PATH, index_col=0).squeeze()
    engine = create_engine('sqlite:////app/database.db')
    table = Table('predictions', MetaData(), autoload_with=engine)
    date_time = datetime.now()
    for session_id, prediction in y_pred.to_dict().items():
        with engine.connect() as connection:
            connection.execute(
                table.insert(), {
                    'session_id': session_id,
                    'prediction': prediction,
                    'date_time': date_time
                }
            )


def process_hits_wrapper() -> None:
    hits = pd.read_csv(INPUT_HITS_PATH, index_col=0)
    utm_source, cars, models, is_action = process_hits(hits)
    utm_source.to_csv(TMP_UTM_SOURCE)
    cars.to_csv(TMP_CARS)
    models.to_csv(TMP_MODELS)
    is_action.to_csv(TMP_IS_ACTION)


def process_sessions_wrapper() -> None:
    sessions = pd.read_csv(INPUT_SESSIONS_PATH, index_col=0)
    utm_source = pd.read_csv(TMP_UTM_SOURCE, index_col=0)
    cars = pd.read_csv(TMP_CARS, index_col=0)
    models = pd.read_csv(TMP_MODELS, index_col=0)
    is_action = pd.read_csv(TMP_IS_ACTION, index_col=0).squeeze()
    sessions['is_action'] = sessions.index.map(is_action)
    sessions_processed = process_sessions(sessions, utm_source, cars, models).drop(columns=COLUMNS_TO_DROP)
    sessions_processed = sessions_processed.drop(columns=['visit_date', 'visit_time'])
    sessions_processed.to_csv(TMP_SESSIONS_PATH)


def make_prediction() -> None:
    sessions_processed = pd.read_csv(TMP_SESSIONS_PATH, index_col=0)
    # sessions_processed = sessions_processed.drop(columns='is_action')
    model = load_model()
    data = process_features(sessions_processed, model)
    y_pred = model.predict(data)
    y_pred = pd.Series(y_pred, index=data.index, name='prediction').astype(int)
    y_pred.to_csv(OUTPUT_PREDICTION_PATH)


dag = DAG(
    'catboost_prediction_pipeline_db',
    description=(
        'Make predictions using a CatBoost model and hits and sessions'
        ' in a pipeline setting and save into a SQLite database'
    ),
    schedule_interval='@once',
    start_date=datetime(2023, 5, 31)
)

process_hits_task = PythonOperator(
    task_id='process_hits',
    python_callable=process_hits_wrapper,
    dag=dag
)

process_sessions_task = PythonOperator(
    task_id='process_sessions',
    python_callable=process_sessions_wrapper,
    dag=dag
)

make_prediction_task = PythonOperator(
    task_id='make_prediction',
    python_callable=make_prediction,
    dag=dag
)

save_prediction_to_db_task = PythonOperator(
    task_id='save_prediction_to_db',
    python_callable=save_prediction_to_db,
    dag=dag
)

process_hits_task >> process_sessions_task >> make_prediction_task >> save_prediction_to_db_task
