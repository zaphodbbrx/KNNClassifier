import os
import click
import pickle
import numpy as np
from src.model import ModelPipeline
from src.data_utils import read_csv_data, train_test_split
from src.metrics import accuracy
from src.conf import *


@click.command()
def build_model():
    """Скрипт для потсроения модели"""
    print('reading/preparing train data')
    raw_data = read_csv_data(train_csv_path)

    assert all([f in raw_data.columns.tolist() for f in feats]), 'not all feature columns are present in train data'
    assert 'y' in raw_data, 'target column is not present in data'
    assert raw_data.shape[0] > N_NEIGHBORS

    X = raw_data[feats].values
    y = raw_data['y'].values

    assert not (X == X[0]).all(), 'all rows in training data are the same'
    assert np.unique(y).shape[0]>1, 'target data must contain more than 1 unique value'

    X_train, y_train, X_val, y_val = train_test_split(X, y, 0.2)

    print('training model')
    mdl = ModelPipeline(steps)

    mdl.fit(X_train, y_train)

    y_pred = mdl.predict(X_val)

    acc = accuracy(y_pred, y_val)

    print(f'validation accuracy: {acc}')

    print('saving model')
    if not os.path.exists(saved_models_path):
        os.makedirs(saved_models_path)

    pickle.dump(mdl, open(os.path.join(saved_models_path, 'mdl.p'), 'wb'))
    print('done')


if __name__ == '__main__':

    build_model()
