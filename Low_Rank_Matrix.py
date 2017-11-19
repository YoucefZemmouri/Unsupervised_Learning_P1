from tools import *

def Pro(W,X):
    """
    :param W: Binary Matrix
    :param X: Data Matrix
    :return: Orthogonal projection onto the span of all matrices
    vanishing outside of W
    """
    Res = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if W[i,j] == 1:
                Res[i,j] = X[i,j]
            else:
                Res[i,j] = 0
    return Res

def SoftThreshold(epsilon,x):
    return np.sign(x)*np.maximum(np.abs(x)-epsilon,0)

def D_operator(tau, X):
    """
    :param tau:
    :param X:
    :return:
    """
    U, s, V = np.linalg.svd(X, full_matrices=True)
    Sigma = np.zeros(X.shape)
    Sigma[:s.shape[0], :s.shape[0]] = np.diag(s)

    return np.dot(U, np.dot(SoftThreshold(tau,Sigma), V))



def LRMC(X,W,tau,beta):
    """
    :param X: Data Matrix
    :param W: Binary Matrix
    :param tau: Parameter of the optimization problem
    :param beta: Step size of the dual gradient ascent step
    :return: Low-rank completion of the matrix X
    """
    if CheckBinary(W):
        Z = np.zeros(X.shape)
        for i in range(100):
            A = D_operator(tau,Pro(W,Z))
            Z = Z + beta*(Pro(W,X)-Pro(W,Z))
        return A
    else:
        print("Matrix W is not binary")

X = np.random.rand(10,5)
W = np.random.rand(10,5)

tau = 0.01
beta = 0.1

A = LRMC(X,W,tau,beta)
print(A)
