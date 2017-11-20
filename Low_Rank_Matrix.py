from tools import *

np.random.seed(123456)

# Creates a matrix whose columns are affinely dependent 
X0 = np.random.randint(10,20, size=5)  # generate in range [10,20], avoiding zero to distinguish easily later
X = []
for i in range(10):
    X.append(list(X0 + np.random.randint(-10,10)))
X = np.array(X)

print("Initial matrix : ")
print(np.around(X,decimals=2))
    
# Choose observed entries at random
gamma = 0.3  # ratio of missing entries
indices = np.random.choice(X.size, int(gamma*X.size), replace=False)  # choose randomly indices to mask
W = np.ones(X.shape)
for i in indices:
    W.itemset(i, 0)  # flatten indexing

print("Observed entries : ")
print(np.around(X*W,decimals=2))

tau = 1000
beta = 0.01
epsilon = 0.1

A = LRMC(X, W, tau, beta, epsilon)

np.set_printoptions(precision=2)
print("Recovered matrix : ")
print(np.around(A,decimals=2))

print("Error matrix : ")
print(np.around(A-X, decimals=2))
