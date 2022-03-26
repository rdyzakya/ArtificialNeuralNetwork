import numpy as np


def sum_squared_error(
    y_true: np.ndarray, y_pred: np.ndarray, derivative: bool = False
) -> float:
    """
    [DESC]
        Function to calculate sum squared error
    [PARAMS]
        y_true : np.ndarray
        y_pred : np.ndarray
        derivative : bool
    [RETURN]
        float
    """
    if derivative:
        return np.sum(y_true - y_pred,axis=0)
    return np.sum(np.square(y_true - y_pred)) / 2

def entropy(y_true:float, y_pred:float,derivative=False):
    """
    [DESC]
        Function to calculate entropy
    [PARAMS]
        y_true : float
        y_pred : float
        derivative : bool
    [RETURN]
        float
    """
    if not derivative:
        if y_true == 1:
            if y_pred == 0:
                return -1
            return -np.log(y_pred)
        else:
            if y_pred == 1:
                return -1
            return -np.log(1-y_pred)
    if y_true == 1:
        if y_pred == 0:
            y_pred += 0.001
        return -1/y_pred
    else:
        if y_pred == 1:
            y_pred -= 0.001
        return 1/(1-y_pred)
        


def cross_entropy_error(
    y_true: np.ndarray, y_pred: np.ndarray, derivative: bool = False, epsilon:float=1e-12) -> float:
    """
    [DESC]
        Function to calculate cross entropy error
    [PARAMS]
        y_true : np.ndarray
        y_pred : np.ndarray
        derivative : bool
        epsilon : float
    [RETURN]
        float
    """
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    # vect = np.vectorize(lambda true, pred, derivative: entropy(true, pred, derivative))
    if derivative:
        # return np.sum(vect(y_true, y_pred, derivative), axis=0)
        return np.sum(-y_true/y_pred + (1-y_true)/(1-y_pred),axis=0)
    N = y_pred.shape[0]
    ce = -np.sum(y_true*np.log(y_pred+1e-9))/N
    return ce