import datetime as dt
import os
import sys

from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

path = os.path.expanduser('~/airflow_hw')
# Добавим путь к коду проекта в переменную окружения, чтобы он был доступен python-процессу
os.environ['PROJECT_PATH'] = path
# Добавим путь к коду проекта в $PATH, чтобы импортировать функции
sys.path.insert(0, path)

from modules.pipeline import pipeline

import logging
from datetime import datetime
import dill
import json
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 10),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}

path = os.environ.get('PROJECT_PATH', '.')


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    return df.drop(columns_to_drop, axis=1)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        bounds = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return bounds

    df = df.copy()
    boundaries = calculate_outliers(df['year'])
    df.loc[df['year'] < boundaries[0], 'year'] = round(boundaries[0])
    df.loc[df['year'] > boundaries[1], 'year'] = round(boundaries[1])
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    def short_model(x):
        if not pd.isna(x):
            return x.lower().split(' ')[0]
        else:
            return x

    df = df.copy()
    df.loc[:, 'short_model'] = df['model'].apply(short_model)
    df.loc[:, 'age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return df


def pipeline() -> None:
    df = pd.read_csv(f'{path}/data/train/homework.csv')

    X = df.drop('price_category', axis=1)
    y = df['price_category']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('outlier_remover', FunctionTransformer(remove_outliers)),
        ('feature_creator', FunctionTransformer(create_features)),
        ('column_transformer', column_transformer)
    ])

    models = [
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    ]

    best_score = .0
    best_pipe = None
    for model in models:

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        logging.info(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    logging.info(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')

    best_pipe.fit(X, y)
    model_filename = f'{path}/data/models/cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl'

    with open(model_filename, 'wb') as file:
        dill.dump(best_pipe, file)

    logging.info(f'Model is saved as {model_filename}')


def load_model(model_file):
    with open(model_file, 'rb') as file:
        model = dill.load(file)
    return model


def make_predictions(folder, model):

    prediction_results = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r') as file:

            data = json.load(file)
            df = pd.DataFrame(data, index=[0])
            prediction = model.predict(df)
            prediction_results.append({'car_id': filename.split('.')[0], 'pred': prediction[0]})

    return prediction_results


def predictions_to_csv(predictions, output_folder):

    predictions_df = pd.DataFrame(predictions)
    output_file = os.path.join(output_folder, 'predictions.csv')
    predictions_df.to_csv(output_file, index=False)


def predict() -> None:
    model_folder = f'{path}/data/models/'
    model_file = os.listdir(model_folder)[0]
    model = load_model(os.path.join(model_folder, model_file))
    predictions = make_predictions(f'{path}/data/test', model)
    predictions_to_csv(predictions, f'{path}/data/predictions')


with DAG(
        dag_id='car_price_prediction',
        schedule_interval="@once",
        default_args=args,
) as dag:

    first_task = BashOperator(
        task_id='first_task',
        bash_command = 'echo "Here we go!"',
        dag=dag,
    )
    pipeline = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
        dag=dag,
    )

    predict = PythonOperator(
        task_id='predict',
        python_callable=predict,
        dag=dag,
    )

    first_task >> pipeline >> predict
    # <YOUR_CODE>
