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
    exps = np.exp(arr - arr.max())
    p =  exps / np.sum(exps, axis=0)
    if(derivative):
        retval = []
        for i in range(len(p)):
                temp = []
                for j in range(len(p)):
                        if(i == j):
                                temp.append(p[i]*(1-p[i]))
                        else:
                                temp.append(-p[i]*p[j])
                retval.append(temp)
        return np.sum(np.array(retval),axis=0)
    return p
