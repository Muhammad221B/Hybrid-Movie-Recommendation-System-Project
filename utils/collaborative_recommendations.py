import joblib
import pandas as pd

ratings = pd.read_csv('../data/ratings.csv') 
movies = pd.read_csv('../data/movies.csv') 

svd = joblib.load("../models/svd_model.pkl")

def collaborative_recommendations(user_id, top_n=10):
    user_rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist() # get User rated_Movies
    movie_ids = movies['movieId'].unique()
    movies_to_predict = [mid for mid in movie_ids if mid not in user_rated_movies]
    predictions = []

    for movie_id in movies_to_predict:
        pred = svd.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_preds = predictions[:top_n]
    movie_ids = [x[0] for x in top_preds]

    return movies[movies['movieId'].isin(movie_ids)][['title', 'movieId']]