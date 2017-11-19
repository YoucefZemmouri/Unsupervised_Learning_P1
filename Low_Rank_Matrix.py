from tools import *

X = np.random.rand(10, 5)
W = np.random.randint(2, size=X.shape)

tau = 0.01
beta = 0.1

A = LRMC(X, W, tau, beta)
print(A)
