import streamlit as st
import pickle
import pandas as pd
import requests
import time
from sklearn.metrics.pairwise import cosine_similarity

movies, tfidf = pickle.load(open("recommender.pkl", "rb"))
tfidf_matrix = tfidf.transform(movies['content'])
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

#Fetch Poster
def fetch_poster(movie_id):
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(
                f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=3bb4dd17e28c934dee13937640e3d32f'
            )
            response.raise_for_status()
            data = response.json()

            if 'poster_path' in data and data['poster_path']:
                return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
            else:
                return "https://via.placeholder.com/500x750.png?text=No+Poster"

        except requests.exceptions.RequestException:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return "https://via.placeholder.com/500x750.png?text=Error"

    return "https://via.placeholder.com/500x750.png?text=Unknown+Error"

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_posters

st.title("🎬 Movie Recommender System")
st.write("Find movies similar to your favorites!")

selected_movie_name = st.selectbox("Choose a movie:", movies['title'].values)

if st.button("Recommend"):
    names, posters = recommend(selected_movie_name)

    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.text(names[idx])
            st.image(posters[idx])
