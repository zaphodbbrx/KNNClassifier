import os
import csv
import click
import pickle

from src.model import ModelPipeline
from src.data_utils import read_csv_data
from src.conf import *


@click.command()
def make_predictions():
    """Скрипт для выполнения предсказаний"""
    if not os.path.exists(os.path.join(saved_models_path, 'mdl.p')):
        raw_data = read_csv_data(train_csv_path)
        assert all([f in raw_data.columns.tolist() for f in feats]), 'not all feature columns are present in train data'
        assert 'y' in raw_data, 'target column is not present in data'
        assert raw_data.shape[0] > N_NEIGHBORS

        X = raw_data[feats].values
        y = raw_data['y'].values
        mdl = ModelPipeline(steps)
        mdl.fit(X, y)
    else:
        mdl = pickle.load(open(os.path.join(saved_models_path, 'mdl.p'), 'rb'))

    predictions = read_csv_data(test_csv_path)

    assert all([f in predictions.columns.tolist() for f in feats]), 'not all feature columns are present in test data'

    X_test = predictions[feats].values

    y_pred = mdl.predict(X_test)

    predictions['y'] = y_pred

    submission_df = read_csv_data(submission_csv).drop('y', axis=1)

    submission = predictions[['ID', 'y']].merge(
        right=submission_df,
        on='ID'
    )

    submission.to_csv('data/submission.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':

    make_predictions()
