from tools import *

# Creates a matrix whose columns are affinely dependent 
X0 = np.random.randint(10, size=5)
X = []
for i in range(10):
    X.append(list(X0 + np.random.randint(-10,10)))
X = np.array(X)

print("Initial matrix : ")
print(np.around(X,decimals=2))
    
# Choose observed entries at random
W = np.random.randint(2, size=X.shape)

print("Observed entries : ")
print(np.around(X*W,decimals=2))

tau = 1000
beta = 0.01
epsilon = 0.01

A = LRMC(X, W, tau, beta, epsilon)

print("Recovered matrix : ")
print(np.around(A,decimals=2))
