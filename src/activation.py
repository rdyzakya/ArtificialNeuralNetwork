import numpy as np
from typing import List


def relu(x: float, derivative: bool = False) -> float:
    """
    [DESC]
            ReLU function
    [PARAMS]
            x : float
            derivative = bool
    [RETURN]
            float
    """
    if derivative:
        if x < 0:
            return 0
        else:
            return 1
    return max(0, x)


def linear(x: float, derivative: bool = False) -> float:
    """
    [DESC]
            Linear function
    [PARAMS]
            x : float
            derivative = bool
    [RETURN]
            float
    """
    if derivative:
        return 1
    return x


def sigmoid(x: float, derivative: bool = False) -> float:
    """
    [DESC]
            Sigmoid function
    [PARAMS]
            x : float
            derivative = bool
    [RETURN]
            float
    """
    p = 1 / (1 + np.exp(-x))
    if derivative:
        return p * (1 - p)
    return p


def softmax(arr: List[float]) -> List[float]:
    """
    [DESC]
            Softmax function
    [PARAMS]
            arr : list of float
    [RETURN]
            list of float
    """
    return np.exp(arr) / np.sum(np.exp(arr))


def softmax_derive(probs: List[float], toDerive: int, respectTo: int) -> float:
    """
    [DESC]
            Softmax function derivation
    [PARAMS]
            probs: list of probabillities
            toDerive: which probability to derive
            respectTo: with respect to which raw value
    [RETURN]
            float, derivation result
    """
    if toDerive == respectTo:
        return probs[toDerive] * (1 - probs[toDerive])
    else:
        return -probs[toDerive] * probs[respectTo]
