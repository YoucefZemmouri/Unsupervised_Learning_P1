import pandas as pd

# We use the database "ml-latest-small" downloaded on the Movielens website

movies = pd.read_csv("movies-data/movies.csv")
GoodMovies_idx = pd.Series(
    [(("Horror" in G) | ("Romance" in G)) for G in movies.genres], 
    index = movies.index
)
GoodMovies = movies[GoodMovies_idx].movieId

ratings = pd.read_csv("movies-data/ratings.csv")

GoodRatings_idx_movies = [m in GoodMovies for m in ratings.movieId]
GoodRatings = ratings[GoodRatings_idx_movies][["userId","movieId","rating"]]

RatingsPerUser = GoodRatings.userId.value_counts()
GoodUsers = RatingsPerUser[RatingsPerUser >= 4].index
GoodRatings_idx_users = [u in GoodUsers for u in GoodRatings.userId]
GoodRatings = GoodRatings[GoodRatings_idx_users]

GoodRatings.to_csv("movies-data/movielens_dataset.csv", index=False)
