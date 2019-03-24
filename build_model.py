import os
import click
import pickle

from src.model import ModelPipeline
from src.data_utils import read_csv_data, train_test_split
from src.metrics import accuracy
from src.conf import *


@click.command()
def build_model():
    """Скрипт для потсроения модели"""
    print('reading/preparing train data')
    raw_data = read_csv_data(train_csv_path)

    X = raw_data[feats].values
    y = raw_data['y'].values

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
