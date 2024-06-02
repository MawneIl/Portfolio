import dill
import os
import pandas as pd
import logging
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# variables:
''' Укажем путь к файлам проекта:
# -> $PROJECT_PATH при запуске в Airflow
# -> иначе - текущая директория при локальном запуске
'''
path = os.environ.get('PROJECT_PATH', '.')

target_actions = ['sub_car_claim_click',
                  'sub_car_claim_submit_click',
                  'sub_open_dialog_click',
                  'sub_custom_question_submit_click',
                  'sub_call_number_click',
                  'sub_callback_submit_click',
                  'sub_submit_success',
                  'sub_car_request_submit_click'
                  ]


logging.basicConfig(level=logging.INFO, filename=f"{path}/logs/project_log.log", filemode="a",
                    format="%(asctime)s %(levelname)s %(message)s")


def df_load():
    files = ['ga_hits.pkl', 'ga_sessions.pkl']
    # загрузка данных
    with open(f'{path}/data/{files[0]}', 'rb') as file:
        df_hits = dill.load(file)

    with open(f'{path}/data/{files[1]}', 'rb') as file:
        df_sessions = dill.load(file)

    # создание списка сессий с определенным целевым событием
    df_hits['target'] = df_hits.event_action.apply(lambda x: 1 if x in target_actions else 0)
    pivot_table = pd.pivot_table(df_hits, index=['session_id'], values=['target'],
                                 aggfunc={'target': [lambda x: 0 if x.sum() == 0 else 1]}
                                 ).reset_index(level=0)
    pivot_table.columns = ['session_id', 'target']

    # объединение данных
    df = df_sessions.join(pivot_table.set_index('session_id'), on='session_id', how='inner')

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    organic_traffic = ['organic', 'referral', '(none)']

    social_media = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt',
                    'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHNBGo',
                    'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm'
                    ]

    df['utm_social_media'] = df.apply(
        lambda x: 1 if x.utm_source in social_media else 0, axis=1)

    df['utm_traffic'] = df.apply(
        lambda x: 'organic' if x.utm_medium in organic_traffic else 'paid', axis=1)

    df['geo_from_russia'] = df.apply(lambda x: 1 if x.geo_country == "Russia" else 0, axis=1)

    df['geo_full'] = df.apply(lambda x: f'{x.geo_country}: {x.geo_city}', axis=1)

    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = {
        'session_id',
        'client_id',
        'visit_date',
        'visit_time',
        'visit_number',
        'utm_keyword',
        'utm_medium',
        'utm_source',
        'device_os',
        'device_model',
        'device_screen_resolution',
        'geo_country',
        'geo_city'
    }

    columns_to_save = list(set(df.columns.to_list()) - columns_to_drop)

    return df[columns_to_save]


def pipeline():
    print('Target event predictor Pipeline')
    logging.info('Target event predictor Pipeline')

    # input data:
    df = df_load()

    df = df[df.utm_source.notna()]

    x = df.drop('target', axis=1)
    y = df['target']

    class_weight = {0: 1, 1: 33}

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant',
                                  fill_value='other')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('feature_creator', FunctionTransformer(create_features)),
        ('filter', FunctionTransformer(filter_data)),
        ('column_transformer', column_transformer)
    ])

    models = [
        LogisticRegression(solver='liblinear', class_weight=class_weight, random_state=42),
        RandomForestClassifier(class_weight=class_weight, n_jobs=-1, oob_score=True, random_state=42)
    ]

    best_score = .0
    best_pipe = None
    for model in models:

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, x, y, cv=4, scoring='roc_auc')
        logging.info(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(x, y)
    accuracy = accuracy_score(best_pipe.predict(x), y)
    logging.info(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, '
                 f'roc_auc: {best_score:.4f}, '
                 f'accuracy: {accuracy: 2f}')

    model_filename = f'{path}/models/best_pipe.pkl'

    with open(model_filename, 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Target event prediction pipeline',
                'autor': 'Nail Mavliev',
                'version': 1.0,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'roc_auc': best_score,
                'accuracy': accuracy
            }
        }, file)

    logging.info(f'Model is saved as {model_filename}')


if __name__ == '__main__':
    pipeline()
