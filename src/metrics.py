import numpy as np


def accuracy(pred: np.array, true: np.array) -> float:
    """Метрика качества (accuracy)"""
    assert true.shape == pred.shape
    return true[true == pred].shape[0] / true.shape[0]
