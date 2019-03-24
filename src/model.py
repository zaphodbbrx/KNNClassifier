from abc import ABC, abstractmethod
from typing import List

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mode


class BaseModel(ABC):
    """Базовый класс для моделей предобработки и классификации"""
    @abstractmethod
    def fit(self, x: np.array, y: np.array) -> None:
        """Настройка модели"""
        pass

    @abstractmethod
    def predict(self, x: np.array) -> np.array:
        """Применение модели"""
        pass


class ModelPipeline(BaseModel):
    """Композит для применения моделей"""
    def __init__(self, steps: List[BaseModel]):

        self.__steps = steps

    def fit(self, x: np.array, y: np.array) -> None:
        tmp = x.copy()
        for step in self.__steps:
            step.fit(tmp, y)
            tmp = step.predict(tmp)

    def predict(self, x: np.array) -> np.array:
        tmp = x.copy()
        for step in self.__steps:
            tmp = step.predict(tmp)
        return tmp


class KnnModel(BaseModel):
    """Классификатор к-ближайших соседей с евклидовой метрикой"""

    def __init__(self, n_neighbors: int):

        self.__n_neighbors = n_neighbors
        self.__x = None
        self.__y = None
        self.__tree = None

    def fit(self, x: np.array, y: np.array) -> None:
        self.__x = x
        self.__y = y
        self.__tree = cKDTree(self.__x)

    def predict(self, x: np.array) -> np.array:
        return np.apply_along_axis(arr=x, axis=1, func1d=self.neighbors_predict)

    def neighbors_predict(self, p):
        """Прогноз для одного примера"""
        nearest_dist, nearest_ind = self.__tree.query(p, k=self.__n_neighbors)
        return mode(self.__y[nearest_ind])[0][0]
