import pandas as pd
import random

movies = pd.read_csv("movies-data/movies.csv")
GoodMovies_idx = pd.Series(
    [(("Horror" in G) | ("Romance" in G)) for G in movies.genres], 
    index = movies.index
)
GoodMovies = movies[GoodMovies_idx].movieId

ratings = pd.read_csv("movies-data/ratings.csv")
GoodRatings_idx = [(m in GoodMovies) for m in ratings.movieId]
GoodRatings = ratings[GoodRatings_idx][["userId","movieId","rating"]]

print("Number of movies :", len(GoodMovies))
print("Number of users :", len(GoodRatings.userId.unique()))
print("Number of ratings :", len(GoodRatings.rating))

percentage = 0.8
train_len = int(0.8*len(GoodRatings.rating))
test_len = len(GoodRatings.index) - train_len

print("Size of training set :", train_len,"(",percentage*100,"% of the total dataset )")
print("Size of test set :", test_len)

RatingsTrain = GoodRatings
RatingsTest = pd.DataFrame()

while RatingsTrain.index.size > train_len:
    test = RatingsTrain.sample()
    movieOccurs = (RatingsTrain.movieId == test.movieId.iloc[0]).sum() > 1
    userOccurs = (RatingsTrain.userId == test.userId.iloc[0]).sum() > 1
    if movieOccurs & userOccurs:
        RatingsTest = RatingsTest.append(test)
        RatingsTrain = RatingsTrain.drop(test.index[0])

print("Verification : Size of training set :", RatingsTrain.index.size)
print("Verification : Size of test set :", RatingsTest.index.size)

RatingsTrain.to_csv("movies-data/Train.csv", index=False)
RatingsTest.to_csv("movies-data/Test.csv", index=False)
