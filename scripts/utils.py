import logging

import pandas as pd
from sklearn.externals import joblib

logger = logging.getLogger('root')


def read_data(*data_files):
    logger.info(f'Reading data from files: {data_files}')

    df = pd.concat(pd.read_parquet(f) for f in data_files)
    logger.info(f'Total data records: {len(df)}')

    # Treating event_id as categorical variable
    df.event_id = df.event_id.astype('str')

    # Converting start to datetime
    df.start = df.start.astype('datetime64[s]')

    return df


def convert_data(df):
    logger.info('Converting data for machine learning model')

    # Add date column
    df['date'] = df.start.dt.date.astype('datetime64[D]')

    # Add column for counting events within a day for a customer
    df['event_count'] = 1

    # One-hot-encode categorical variables
    categorical = ['event_id', 'event_result', 'cause_code', 'sub_cause_code', 'mecontext']
    other = ['ident', 'date', 'event_count']
    if 'probs' in df.columns:
        other.append('probs')
    logger.info(f'Categorical features: {categorical}')
    logger.info(f'Quantitative features: {other}')
    df = pd.concat([df[other], pd.get_dummies(df[categorical])], axis=1)

    # Group by customer days
    df = df.groupby(['ident', 'date']).sum()

    if 'probs' in df.columns:
        df.probs = df.probs > 0
        logger.info(f'Positive examples {(df.probs == True).sum()}')
        logger.info(f'Negative examples {(df.probs == False).sum()}')

    logger.info(f'Total converted data records: {len(df)}')

    return df


def save_model(model, filename):
    logger.info(f'Saving model to {filename}')
    joblib.dump(model, filename)


def load_model(filename):
    logger.info(f'Loading model form {filename}')
    return joblib.load(filename)
