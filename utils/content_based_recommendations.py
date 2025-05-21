import numpy as np
import pandas as pd 
from preprocess_title import preprocess_title


movies = pd.read_csv('../data/movies.csv')   

cosine_sim = np.load("../models/cosine_sim.npy")

def content_based_recommendations(movie_title, top_n=10):
    movie_title_clean = preprocess_title(movie_title)

    
    if movie_title_clean not in movies['clean_name'].values:
        print(f"Movie '{movie_title}' not found in dataset.")
        return pd.DataFrame()


    idx = movies[movies['clean_name'] == movie_title_clean].index[0] # Movie_ID
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1] # first Movie is the same movie
    movie_indices = [i[0] for i in sim_scores] # Get Movie_ID


    return movies.iloc[movie_indices][['title', 'movieId']]