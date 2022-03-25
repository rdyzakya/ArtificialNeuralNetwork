import numpy as np


def init_weight(row: int, column: int) -> np.ndarray:
    """
    [DESC]
            Function to initialize weights with dimension = m x n
    [PARAMS]
            row : int
            column : int
    [RETURN]
            numpy.ndarray(float)
    """
    return np.random.randn(row, column)


def init_bias(n: int) -> np.ndarray:
    """
    [DESC]
            Function to initialize biases with length = n
    [PARAMS]
            n : int
    [RETURN]
            numpy.ndarray(float)
    """
    return np.zeros(n)
