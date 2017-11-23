import pandas as pd

dataset = "original" # If we use the file provided by Réne Vidal
#dataset = "movielens" # If we use the Movielens database we built with P3_movielens.py

GoodRatings = pd.read_csv("movies-data/"+dataset+"_dataset.csv")

print("Number of movies :", len(GoodRatings.movieId.unique()))
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

RatingsTrain.to_csv("movies-data/"+dataset+"_Train.csv", index=False)
RatingsTest.to_csv("movies-data/"+dataset+"_Test.csv", index=False)
