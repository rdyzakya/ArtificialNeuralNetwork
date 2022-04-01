import numpy as np

epsilon = 1e-12

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
    return np.sum(np.square(y_true - y_pred)) / 2 # not axis 0 because this is used only for model.errors
        


def cross_entropy_error(
    y_true: np.ndarray, y_pred: np.ndarray, derivative: bool = False) -> float:
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
    if derivative:
        return np.sum(-y_true/y_pred + (1-y_true)/(1-y_pred),axis=0)
    ce = np.sum(-y_true*np.log(y_pred) - (1-y_true)*np.log(1-y_pred)) # not axis 0 because this is used only for model.errors
    return ce