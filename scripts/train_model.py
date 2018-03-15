#!/usr/bin/env python

import logging
import os
import sys

from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

sys.path.append(Path(__file__).parent)
from utils import read_data, convert_data, save_model


MODEL_PATH = Path(__file__).parents[1].joinpath('models')

SESSION_NAME = f'{datetime.now():%Y-%m-%d_%H%M%S}'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('root')
logger.addHandler(
    logging.FileHandler(MODEL_PATH.joinpath(SESSION_NAME + '.log'))
)

SEED = 1503
np.random.seed(SEED)


def get_session_name():
    now = datetime.now()

    return f'{now.date().isoformat()}_{now.hour*60 + now.minute}'


def train_model(df):
    logger.info('Starting model training')

    feature_names = list(df.columns.drop('probs'))
    logger.info(f'Available features: {feature_names}')

    logger.info('Performing search for optimal training parameters')
    estimator = lgb.LGBMClassifier()
    logger.info(f'Using estimator {estimator}')

    param_grid = {
        'boosting_type': ['gbdt'],
        'class_weight': ['balanced'],
        'colsample_bytree': [1.0],
        'learning_rate': [0.1],
        'n_estimators': [80, 90, 100],
        'num_leaves': [31],
        'reg_alpha': [0.],
        'reg_lambda': [0.2, 0.3],
    }
    logger.info(f'Optimizing parameters: {param_grid}')

    grid = GridSearchCV(estimator, param_grid, n_jobs=-1)

    grid.fit(df[feature_names], df.probs)

    logger.info(f'Best params: {grid.best_params_}')
    logger.info(f'Best score: {grid.best_score_}')

    params = grid.best_params_

    # Fit k models on different training/validation splits
    k = 5

    models = []
    results = []

    for i in range(k):
        logger.info(f'Fitting model {k}')

        X_train, X_test, y_train, y_test = train_test_split(df[feature_names], df.probs, test_size=0.2)

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, feature_name=feature_names)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        result = {
            'train_accuracy_score': accuracy_score(y_train, y_train_pred),
            'train_confusion_matrix': confusion_matrix(y_train, y_train_pred),
            'test_accuracy_score': accuracy_score(y_test, y_test_pred),
            'test_confusion_matrix': confusion_matrix(y_test, y_test_pred),
        }

        logger.info(''.join((
            'Train accuracy {train_accuracy_score}\n',
            'Train confusion matrix \n{train_confusion_matrix}\n',
            'Test accuracy {test_accuracy_score}\n',
            'Test confusion matrix \n{test_confusion_matrix}'
        )).format(**result))

        model.feature_names = feature_names

        models.append(model)
        results.append(result)

    results = pd.DataFrame(results)
    logger.info(f'Average results:\n{results.mean()}')
    logger.info(f'Standard deviation of results:\n{results.std()}')

    return models


def main(*data_files):
    if len(data_files) < 1:
        raise TypeError(f'No data files found! Usage: python {os.path.basename(__file__)} data_file [data_file]...')

    start = datetime.now()
    logger.info(f'Starting {start}')

    df = read_data(*data_files)

    df = convert_data(df)

    model = train_model(df)

    save_model(model=model, filename=MODEL_PATH.joinpath(SESSION_NAME + '.pkl'))

    end = datetime.now()
    logger.info(f'All done {end}, elapsed: {end - start}')


if __name__ == '__main__':
    env_data_files = [f.strip() for f in os.environ.get('TRAIN_FILES', '').split(',') if f.strip()]

    main(*env_data_files, *sys.argv[1:])
