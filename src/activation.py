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


def softmax_derive(x):
        return x * (1-x)

def softmax(arr: List[float], derivative: bool = False) -> List[float]:
    """
    [DESC]
            Softmax function
    [PARAMS]
            arr : list of float
    [RETURN]
            list of float
    """
    if(derivative):
        retval = []
        for i in range(len(arr)):
                temp = []
                for j in range(len(arr)):
                        if(i == j):
                                temp.append(arr[i]*(1-arr[i]))
                        else:
                                temp.append(-arr[i]*arr[j])
                retval.append(temp)
        return np.array(retval)
    return np.exp(arr) / np.sum(np.exp(arr))
