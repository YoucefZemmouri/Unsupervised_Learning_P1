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
    if np.array_equal(Values, W0) or np.array_equal(Values, W1) or np.array_equal(Values, W10):
        return True
    else:
        return False


def CheckShape(X, Y):
    """
    :param X: Matrix
    :param Y: Matrix
    :return: True if X and Y have the same shape, False otherwise
    """
    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        return False
    else:
        return True
    return


def Pro(W, X):
    """
    :param W: Binary Matrix
    :param X: Data Matrix
    :return: Orthogonal projection onto the span of all matrices
    vanishing outside of W
    """
    if CheckBinary(W) and CheckShape(X, W):
        Res = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if W[i, j] == 1:
                    Res[i, j] = X[i, j]
                else:
                    Res[i, j] = 0
        return Res
    else:
        print("Matrix W is not binary or X and W do not have the same shape")


def SoftThreshold(epsilon, x):
    """
    :param epsilon: margin
    :param x: value
    :return: Soft thresholding operator
    """
    return np.sign(x) * np.maximum(np.abs(x) - epsilon, 0)


def D_operator(tau, X):
    """
    :param tau: margin
    :param X: Data Matrix
    :return: Singular value thresholding operator
    """
    U, s, V = np.linalg.svd(X, full_matrices=True)
    Sigma = np.zeros(X.shape)
    Sigma[:s.shape[0], :s.shape[0]] = np.diag(s)

    return np.dot(U, np.dot(SoftThreshold(tau, Sigma), V))


def LRMC(X, W, tau, beta):
    """
    :param X: Data Matrix
    :param W: Binary Matrix
    :param tau: Parameter of the optimization problem
    :param beta: Step size of the dual gradient ascent step
    :return: Low-rank completion of the matrix X
    """
    if CheckBinary(W) and CheckShape(X, W):
        Z = np.zeros(X.shape)
        Z_prev = Z
        A = D_operator(tau, Pro(W, Z))
        Z = Z + beta * (Pro(W, X) - Pro(W, Z))
        while not np.array_equal(Z_prev, Z):
            Z_prev = Z
            A = D_operator(tau, Pro(W, Z))
            Z = Z + beta * (Pro(W, X) - Pro(W, Z))
        return A
    else:
        print("Matrix W is not binary or X and W do not have the same shape")
