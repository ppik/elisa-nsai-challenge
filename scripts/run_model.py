#!/usr/bin/env python

import logging
import os
import sys

from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.append(Path(__file__).parent)
from utils import read_data, convert_data, load_model


MODEL_PATH = Path(__file__).parents[1].joinpath('models')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('root')


def find_latest_model():
    return sorted(MODEL_PATH.glob('*.pkl'), key=os.path.getmtime, reverse=True)[0]


def run_models(model_ensemble, data_file):
    feature_names = model_ensemble[0].feature_names

    df = read_data(data_file)
    df = convert_data(df)

    # Drop extra columns
    extra_columns = set(df.columns) - set(feature_names)
    logging.info(f'Columns present in now, but not in training {extra_columns}')
    df.drop(extra_columns, axis=1, inplace=True)

    # Add missing columns
    missing_columns = set(feature_names) - set(df.columns)
    logging.info(f'Columns present in training, but not in test {missing_columns}')
    for col in missing_columns:
        df[col] = 0

    df = df[feature_names]

    preds = pd.DataFrame(model.predict(df) for model in model_ensemble)

    result = pd.DataFrame()
    result['ident'] = df.reset_index()['ident']
    result['probs'] = preds.mean() > 0.5

    return result


def get_pred_filename(model_file, data_file):
    model_file = Path(model_file)
    data_file = Path(data_file)

    filename = MODEL_PATH.joinpath(
        f"{model_file.name.rsplit('.', 1)[0]}-{data_file.name.rsplit('.', 1)[0]}.csv",
    )
    return filename


def save_preds(preds, filename):
    logger.info(f'Saving inference results to {filename}')
    preds.to_csv(filename, index=False)


def main(model_file, data_files):
    start = datetime.now()
    logger.info(f'Starting {start}')
    model_ensemble = load_model(model_file)

    for data_file in data_files:
        preds = run_models(model_ensemble, data_file)

        filename = get_pred_filename(model_file, data_file)
        save_preds(preds, filename)

    end = datetime.now()
    logger.info(f'All done {end}, elapsed: {end - start}')


if __name__ == '__main__':
    env_data_files = [f.strip() for f in os.environ.get('RUN_FILES', '').split(',') if f.strip()]

    if len(sys.argv) < 2:
        model_file = find_latest_model()
        arg_data_files = sys.argv[1:]
    else:
        model_file = sys.argv[1]
        arg_data_files = sys.argv[2:]

    main(model_file=model_file, data_files=(*env_data_files, *arg_data_files))
