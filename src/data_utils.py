from typing import Tuple

import pandas as pd
import numpy as np

from src.model import BaseModel


def read_csv_data(csv_path: str) -> pd.DataFrame:
    """Считывает данные из csv-файла"""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        raise ValueError('csv file seems to have wrong format')
    return df


def train_test_split(x: np.array, y: np.array, test_size: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """Получение тренировочной и валидационной выборок"""
    assert x.shape[0] == y.shape[0], 'shapes of X and y must be equal'
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    test_size = int(x.shape[0]*test_size)
    train_idx = idx[-test_size:]
    test_idx = idx[:-test_size]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


class ColumnScaler(BaseModel):
    """Модель для масштабирования признаков"""
    def __init__(self, value_range: tuple):
        self.__range = value_range
        self.__a = None
        self.__b = None

    def fit(self, x: np.array, y: np.array = None):

        self.__a = np.min(x, axis=0)
        self.__b = np.max(x, axis=0)

    def predict(self, x: np.array) -> np.array:

        return (self.__range[1] - self.__range[0]) * (x-self.__a) / (self.__b-self.__a) + self.__range[0]
