import json
from pathlib import Path

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from ml_cars.constants import CAT_COLUMNS
from ml_cars.make_features import get_features


def train(sessions: str, hits: str, test_size: float = 0.2, random_state: int = 42,
          output_dir: str = 'models', model_name: str = 'last', **kwargs) -> CatBoostClassifier:
    X, y = get_features(sessions, hits, return_Xy=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = CatBoostClassifier(
        random_seed=random_state,
        max_depth=kwargs.get('max_depth', 3),
        n_estimators=kwargs.get('n_estimators', 100),
        cat_features=CAT_COLUMNS,
        auto_class_weights='Balanced'
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        verbose=10,
    )

    scores = {
        'f1_train': f1_score(y_train, clf.predict(X_train)),
        'f1_test': f1_score(y_test, clf.predict(X_test))
    }

    clf = CatBoostClassifier(
        random_seed=random_state,
        max_depth=kwargs.get('max_depth', 3),
        n_estimators=clf.best_iteration_+1,
        cat_features=CAT_COLUMNS,
        auto_class_weights='Balanced'
    )
    clf.fit(X, y, verbose=100)

    clf.save_model(Path(output_dir, model_name+'.cbm'))
    with open(Path(output_dir, model_name+'.json'), 'w') as file:
        json.dump(scores, file)

    return clf
