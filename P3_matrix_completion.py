import pandas as pd
import numpy as np
from tools import LRMC

def ratingsToMatrix(R,rows,cols):
    X = pd.DataFrame(
        index=rows,
        columns=cols
    )
    for user in R.userId.unique():
        RUser = R[R.userId == user]
        RUser.index = RUser.movieId
        X[user] = RUser.rating

    W = 1-X.isnull()
    X = X.fillna(0).as_matrix()
    W = W.as_matrix()

    return X, W

RatingsTrain = pd.read_csv("movies-data/Train.csv")
RatingsTest = pd.read_csv("movies-data/Test.csv")

# RatingsTrain = pd.read_csv("movies-data/specific_Train.csv")
# RatingsTest = pd.read_csv("movies-data/specific_Test.csv")

rows = RatingsTrain.movieId.unique()
cols = RatingsTrain.userId.unique()

[X, W] = ratingsToMatrix(RatingsTrain, rows, cols)
[X_test, W_test] = ratingsToMatrix(RatingsTest, rows, cols)

D,N = X.shape
M = np.sum(W)

tau = 1000
beta = min(2,D*N/M)
epsilon = 0.1

A = LRMC(X,W,tau,beta,epsilon)

print(
    "Mean prediction error :",
    np.sum(abs(A*W_test - X_test))/RatingsTest.index.size,
    "\nMean squared prediction error :",
    np.sum((A*W_test - X_test)**2)/RatingsTest.index.size,
    "\nMaximal prediction error :",
    np.max(abs(A*W_test - X_test))
)
