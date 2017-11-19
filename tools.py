import numpy as np


def CheckBinary(W):
    """
    :param W: Matrix
    :return: True if the matrix is Binary, False otherwise
    """
    return set(np.unique(W)).issubset([0,1])


def CheckShape(X, Y):
    """
    :param X: Matrix
    :param Y: Matrix
    :return: True if X and Y have the same shape, False otherwise
    """
    return X.shape == Y.shape


def Pro(W, X):
    """
    :param W: Binary Matrix
    :param X: Data Matrix
    :return: Orthogonal projection onto the span of all matrices
    vanishing outside of W
    """
    if CheckBinary(W) and CheckShape(X, W):
        return W*X
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


def LRMC(X, W, tau, beta, epsilon):
    """
    :param X: Data Matrix
    :param W: Binary Matrix
    :param tau: Parameter of the optimization problem
    :param beta: Step size of the dual gradient ascent step
    :param epsilon: Controls the convergence of the algorithm
    :return: Low-rank completion of the matrix X
    """
    if CheckBinary(W) and CheckShape(X, W):
        Z = np.zeros(X.shape)
        Z_prev = Z+1 # Set it to an arbitrery value but different from Z
        Pro_WX = Pro(W, X) # To avoid calculating it at each iteration
        while sum(sum(abs(Z-Z_prev))) > epsilon:
            Z_prev = Z
            A = D_operator(tau, Pro(W, Z))
            Z = Z + beta * (Pro_WX - Pro(W, A))
        return A
    else:
        print("Matrix W is not binary or X and W do not have the same shape")
