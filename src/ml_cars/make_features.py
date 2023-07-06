import re

import numpy as np
import pandas as pd

from typing import Tuple
from ml_cars.constants import CAT_COLUMNS


CAT_FILL_VALUE = '(none)'
ACTIONS = [
    'sub_car_claim_click',
    'sub_car_claim_submit_click',
    'sub_open_dialog_click',
    'sub_custom_question_submit_click',
    'sub_call_number_click',
    'sub_callback_submit_click',
    'sub_submit_success',
    'sub_car_request_submit_click'
]

COLUMNS_TO_DROP = ['client_id']


def process_hits(hits: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:

    def utm_list_create(x, list_to_comp):
        r = re.compile(x)
        new_list = list(filter(r.match, list_to_comp))
        new_list = '|'.join(new_list)
        return new_list

    def add_utm(hit_page_path, to_split):
        return (
            hit_page_path
                .str.split('?').str[1]
                .str.split(to_split).str[1]
                .str.split('&').str[0]
        ).replace('', np.nan)


    # Создаем колонку с автомобилями
    hits['cars'] = (
        hits['hit_page_path']
        .str.split('?').str[0]
        .str.split('cars').str[1]
        .str.split('all').str[1]
        .str.findall(r'([\w-]+[\/-][\w-]+)').str.join("")
    )
    hits.loc[hits['cars'] == '', 'cars'] = np.nan

    # Создаем колонки с UTM метками
    hits['list_utm'] = (
        hits['hit_page_path']
        .str.split('?').str[1]
        .str.findall(r'(?:\w\w+=)')
    )
    utm_set = list(set().union(*hits['list_utm'].dropna()))
    utm_tags = [
        'utm_source',
        'utm_medium',
        #'utm_campaign',
        #'utm_term',
        #'utm_content'
        ]
    for col_name in utm_tags:
        hits[col_name] = add_utm(hits['hit_page_path'], utm_list_create(col_name, utm_set)).astype('category')

    # Создаем бинарную колонку, яляется ли запрос действием или нет
    hits['is_action'] = hits['event_action'].isin(ACTIONS).astype(np.uint8)

     # Обратботка колонки utm_source
    utm_source = pd.pivot_table(hits, columns='utm_source', values='is_action', index='session_id', aggfunc='mean', fill_value=0)
    utm_source = utm_source.loc[:, utm_source.sum() > 0]
    utm_source['google'] = (
                            utm_source.get('google', pd.Series(0, index=utm_source.index)) +
                            utm_source.get('google_poisk', pd.Series(0, index=utm_source.index)) +
                            utm_source.get('google_poisk_web', pd.Series(0, index=utm_source.index))
    )

    utm_source = utm_source.loc[:, utm_source.sum() > 1]
    utm_source = utm_source.drop(
        columns=['google_poisk', 'google_poisk_web', 'yandex_poisk', 'yandex_poisk_web', 'dzen', 'vk', 'fb'],
        errors='ignore'
    )

    # Обратботка колонки cars
    cars = pd.pivot_table(hits, columns='cars', values='is_action', index='session_id', aggfunc='mean', fill_value=0)

    # Добавим модель автомобиля
    hits['models'] = hits['cars'].apply(lambda x: x.split('/')[0] if isinstance(x, str) else x)
    models = pd.pivot_table(hits, columns='models', values='is_action', index='session_id', aggfunc='mean', fill_value=0)

    is_action = hits.groupby('session_id')['is_action'].any().astype('int8')

    return utm_source, cars, models, is_action


def process_sessions(sessions: pd.DataFrame, utm_source: pd.DataFrame, cars: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:

    utm_source = utm_source.reindex(sessions['session_id']).fillna(0)
    cars = cars.reindex(sessions['session_id']).fillna(0)
    models = models.reindex(sessions['session_id']).fillna(0)

    sessions = sessions.merge(utm_source, left_on='session_id', right_index=True, how='left')
    sessions = sessions.merge(cars, left_on='session_id', right_index=True, how='left')
    sessions = sessions.merge(models, left_on='session_id', right_index=True, how='left')

    sessions[CAT_COLUMNS] = sessions[CAT_COLUMNS].fillna(CAT_FILL_VALUE)

    return sessions.set_index('session_id')


def get_features(sessions: pd.DataFrame, hits: pd.DataFrame, return_Xy=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    utm_source, cars, models, is_action = process_hits(hits)
    sessions['is_action'] = sessions['session_id'].map(is_action)
    sessions_processed = process_sessions(sessions, utm_source, cars, models).drop(columns=COLUMNS_TO_DROP)
    sessions_processed = sessions_processed.drop(columns=['visit_date', 'visit_time'])

    if return_Xy:
        return sessions_processed.drop(columns='is_action'), sessions_processed['is_action']
    # sessions_processed = sessions_processed.drop(columns='is_action', errors='ignore')
    return sessions_processed
