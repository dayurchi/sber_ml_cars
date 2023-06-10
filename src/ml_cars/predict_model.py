from importlib.resources import path

import pandas as pd
from catboost import CatBoostClassifier

from ml_cars.make_features import get_features


def load_model(model_fine_name='last.cbm'):
    clf = CatBoostClassifier()
    with path('ml_cars.models', model_fine_name) as model_path:
        model = clf.load_model(model_path)
    return model


def process_features(data, model):
    empty_cols = set(model.feature_names_).difference(data.columns)
    data = data.merge(
        pd.DataFrame(0, index=data.index, columns=list(empty_cols)),
        left_index=True,
        right_index=True
    )
    return data


def predict(sessions, hits, model_fine_name='last.cbm'):
    model = load_model(model_fine_name)
    data = get_features(sessions, hits)
    data = process_features(data, model)
    # X_pred = X_pred.drop(columns=['visit_date', 'visit_time'])
    y_pred = model.predict(data)
    y_pred = pd.Series(y_pred, index=data.index, name='prediction').astype(int)
    return y_pred
