import numpy as np
import time

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
    #if CheckBinary(W) and CheckShape(X, W):
    return W*X
    #else:
        #print("Matrix W is not binary or X and W do not have the same shape")

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
    U, s, V = np.linalg.svd(X, full_matrices=False)
    Sigma = np.diag(SoftThreshold(tau, s))

    return np.dot(U, np.dot(Sigma, V))


def LRMC(X, W, tau, beta, epsilon, A_start = None):
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
        Pro_WX = W * X # To avoid calculating it at each iteration
        count = 0
        t = time.time()
        if A_start is not None: # initialize A if given
            Z = Pro_WX - W * A_start
        while True:
            A = D_operator(tau, W * Z)
            dLdZ = Pro_WX - W * A  # gradient of L w.r.t Z
            max_norm = np.max(np.abs(dLdZ))
            print('i=',count, ' step = ', max_norm)
            if max_norm < epsilon:
                break
            Z += beta * dLdZ
            count += 1
        elapsed = time.time()-t
        print(count, ' iterations used, ', elapsed, ' seconds')
        return A
    else:
        print("Matrix W is not binary or X and W do not have the same shape")
