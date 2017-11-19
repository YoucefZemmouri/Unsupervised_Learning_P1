from tools import *


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


X = np.random.rand(10, 5)
W = np.random.randint(2, size=X.shape)

tau = 0.01
beta = 0.1

A = LRMC(X, W, tau, beta)
print(A)
