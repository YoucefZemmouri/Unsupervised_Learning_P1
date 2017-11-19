import numpy as np

def CheckBinary(W):
    """
    :param W: Matrix
    :return: True if the matrix is Binary, False otherwise
    """
    W0 = np.array([0])
    W1 = np.array([1])
    W10 = np.array([0, 1])
    Values = np.unique(W)
    if np.array_equal(Values,W0) or np.array_equal(Values,W1) or np.array_equal(Values,W10):
        return True
    else:
        return False