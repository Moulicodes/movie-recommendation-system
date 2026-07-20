import numpy as np
import pickle
import requests
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

st.set_page_config(
    page_title="Movie Recommender", page_icon="🎬", layout="wide"
)

TMDB_API_KEY = "6ebd2ae5462139b46c7cdd9ee2138e9a"

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get("poster_path")

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster+Available"
    except Exception:
        return "https://via.placeholder.com/500x750?text=Error+Loading+Poster"


@st.cache_data
def load_data_and_matrix():
    # 1. Load just the light movies dataframe (~2-5 MB)
    df = pickle.load(open("movies_df.pkl", "rb"))
    
    # 2. Recompute vectors & similarity matrix on startup
    count_vec = CountVectorizer(stop_words='english')
    count_mat = count_vec.fit_transform(df['cv_col'])
    
    tfidf_vec = TfidfVectorizer(stop_words='english')
    tfidf_mat = tfidf_vec.fit_transform(df['tfidf_col'])
    
    combined_mat = hstack([count_mat, tfidf_mat])
    hybrid_sim = cosine_similarity(combined_mat, combined_mat)
    
    return df, hybrid_sim

df, hybrid_sim = load_data_and_matrix()

def recommend_movies(movie_title, top_n=5):
    movie_index = df[df["title"] == movie_title].index[0]
    similarity_scores = hybrid_sim[movie_index]

    sorted_indices = np.argsort(similarity_scores)[::-1][1 : top_n + 1]

    recommended_titles = []
    recommended_posters = []

    for idx in sorted_indices:
        movie_id = df.iloc[idx].get("id", df.iloc[idx].get("movie_id"))
        recommended_titles.append(df.iloc[idx]["title"])
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_titles, recommended_posters


st.title("🎬 Movie Recommendation System")
st.write(
    "Select a movie to get top recommendations along with official posters!"
)

movie_list = df["title"].values
selected_movie = st.selectbox(
    "Type or select a movie:", movie_list, index=None
)

top_n = st.slider(
    "How many recommendations would you like?",
    min_value=1,
    max_value=10,
    value=5,
)

if st.button("Recommend", type="primary"):
    if selected_movie:
        with st.spinner("⏳ Loading recommendations... Please wait!"):
            names, posters = recommend_movies(selected_movie, top_n=top_n)

        num_cols = min(top_n, 5)
        cols = st.columns(num_cols)

        for i in range(top_n):
            col_idx = i % num_cols
            with cols[col_idx]:
                st.subheader(f"#{i+1}")
                st.write(names[i])
                st.image(posters[i])
    else:
        st.warning("Please select a movie first!")