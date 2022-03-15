import numpy as np

def sum_squared_error(y_true : np.ndarray,y_pred : np.ndarray,derivative : bool=False) -> float:
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
        return np.sum(y_pred - y_true)
    return np.sum(np.square(y_true-y_pred))/2

def cross_entropy_error(y_true : np.ndarray,y_pred : np.ndarray,derivative : bool=False) -> float:
    """
    [DESC]
        Function to calculate cross entropy error
    [PARAMS]
        y_true : np.ndarray
        y_pred : np.ndarray
        derivative : bool
    [RETURN]
        float
    """
    if derivative:
        return -np.sum(y_true/y_pred)
    return -np.sum(y_true*np.log(y_pred))