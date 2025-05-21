from preprocess_title import preprocess_title
from content_based_recommendations import content_based_recommendations
import numpy as np
import joblib
import pandas as pd

ratings = pd.read_csv('../data/ratings.csv') 
movies = pd.read_csv('../data/movies.csv') 

cosine_sim = np.load("../models/cosine_sim.npy")
svd = joblib.load("../models/svd_model.pkl")

def hybrid_recommendations(user_id, movie_title, top_n=10, content_weight=0.5, collab_weight=0.5):
    content_recs = content_based_recommendations(movie_title, top_n=50)

    if content_recs.empty:
        return pd.DataFrame()

    movie_ids = content_recs['movieId'].tolist()
    collab_scores = []

    for movie_id in movie_ids:
        pred = svd.predict(user_id, movie_id)
        collab_scores.append(pred.est)
    
    collab_scores_norm = (np.array(collab_scores) - np.min(collab_scores)) / (np.ptp(collab_scores) + 1e-8)
    content_scores_norm = (np.array([cosine_sim[movies[movies['clean_name']==preprocess_title(movie_title)].index[0], movies[movies['movieId']==mid].index[0]] for mid in movie_ids]) - 0) / 1
    hybrid_scores = content_weight*content_scores_norm + collab_weight*collab_scores_norm
    
    rec_df = content_recs.copy()
    rec_df['score'] = hybrid_scores
    rec_df = rec_df.sort_values(by='score', ascending=False).head(top_n)
    
    return rec_df[['title', 'movieId']]