import pandas as pd
import numpy as np
from tools import LRMC

RatingsTrain = pd.read_csv("movies-data/Train.csv")
RatingsTest = pd.read_csv("movies-data/Test.csv")

X = pd.DataFrame(
	index=RatingsTrain.movieId.unique(),
	columns=RatingsTrain.userId.unique()
)
for user in RatingsTrain.userId.unique():
    RatingsUser = RatingsTrain[RatingsTrain.userId == user]
    RatingsUser.index = RatingsUser.movieId
    X[user] = RatingsUser.rating

W = 1-X.isnull()
X = X.fillna(0).as_matrix()
W = W.as_matrix()

X_test = pd.DataFrame(
	index=RatingsTrain.movieId.unique(),
	columns=RatingsTrain.userId.unique()
)
for user in RatingsTest.userId.unique():
    RatingsUser = RatingsTest[RatingsTest.userId == user]
    RatingsUser.index = RatingsUser.movieId
    X_test[user] = RatingsUser.rating

W_test = 1-X_test.isnull()
X_test = X_test.fillna(0).as_matrix()
W_test = W_test.as_matrix()

tau = 1000
# Leman: beta should be strictly < 2, beta = 2 may diverge, no idea why ...
beta = 1.8
epsilon = 1  # bad precision, just to make the code run fast

A = LRMC(X,W,tau,beta,epsilon)

print(
	"Mean prediction error (on the whole test set) :",
	np.sum(abs(A*W_test - X_test))/RatingsTest.index.size
)
print(
	"Maximal prediction error :",
	np.max(abs(A*W_test - X_test))
)
