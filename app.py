import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import re

def preprocess_title(title):
    title = re.sub(r'[.,]', '', title)
    title = title.strip()
    title = title.lower()
    return title

# ----------------------------
# Load Data & Model
# ----------------------------
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

movies['clean_name'] = movies['title'].apply(preprocess_title)

cosine_sim = np.load("models/cosine_sim.npy") 
svd = joblib.load("models/svd_model.pkl")

# Build reverse index
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# ----------------------------
# Recommendation Functions
# ----------------------------

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
# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎬 Hybrid Movie Recommendation System")

user_id = st.number_input("Enter your User ID", min_value=1, step=1)
movie_title = st.selectbox("Choose a movie you like", sorted(movies["title"].unique()))

if st.button("Get Recommendations"):
    recommendations = hybrid_recommendations(user_id, movie_title)
    if recommendations.empty:
        st.warning("Movie not found in database.")
    else:
        st.success("Top Recommendations:")
        st.dataframe(recommendations)