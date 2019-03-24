from model import KnnModel
from data_utils import ColumnScaler

__all__ = ['steps', 'feats', 'train_csv_path', 'test_csv_path', 'submission_csv', 'saved_models_path']


N_NEIGHBORS = 11

steps = [
    ColumnScaler(value_range=(0, 1)),
    KnnModel(n_neighbors=N_NEIGHBORS)
]

feats = [
    'x1',
    'x2',
    'x3',
    'x4',
    'x5'
]

train_csv_path = 'data/train.csv'
test_csv_path = 'data/test.csv'
submission_csv = 'data/sample_submission.csv'

saved_models_path = 'trained_model'
