import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the dataset
@st.cache_data
def load_movies():
    # Ensure the file is located in the same directory as the app or provide the full path
    movies = pd.read_csv('u.item', sep='|', encoding='ISO-8859-1',
                         names=['movie_id', 'title', 'release_date', 'video_release',
                                'IMDb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                                'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    # Select relevant columns for the recommendation system
    movies = movies[['movie_id', 'title', 'Action', 'Adventure', 'Animation', 'Children',
                     'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                     'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                     'War', 'Western']]
    return movies

# Initialize data and compute similarity matrix
movies = load_movies()
genre_matrix = np.nan_to_num(movies.iloc[:, 2:].values, nan=0.0)
similarity_matrix = cosine_similarity(genre_matrix)

# Recommendation function
def recommend_movies(genre, top_n=5):
    """Recommend top N movies based on the specified genre."""
    if genre not in movies.columns[2:]:  # Validate input genre
        st.error(f"Genre '{genre}' not found! Choose from: {list(movies.columns[2:])}")
        return pd.DataFrame()  # Return empty DataFrame if genre is invalid

    # Filter movies that match the specified genre
    matching_movies = movies[movies[genre] == 1]
    movie_indices = matching_movies.index  # Get indices of matching movies

    # Compute similarity scores for matching movies
    genre_similarities = similarity_matrix[movie_indices][:, movie_indices]

    # Create a DataFrame with similarity scores
    recommendations = matching_movies.copy()
    recommendations['similarity_score'] = genre_similarities.mean(axis=1)

    # Sort movies by similarity score and return top N recommendations
    recommendations = recommendations.sort_values(by='similarity_score', ascending=False)
    return recommendations[['title', 'similarity_score']].head(top_n)

# Streamlit UI
st.title("Movie Recommendation System")

# Genre selection dropdown
genre_options = movies.columns[2:]
selected_genre = st.selectbox("Select a Genre", options=genre_options)

# Movie count slider
num_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)

# Display recommendations when the button is clicked
if st.button("Get Recommendations"):
    recommended_movies = recommend_movies(selected_genre, num_recommendations)
    if not recommended_movies.empty:
        st.write(f"Top {num_recommendations} {selected_genre} Movies:")
        st.dataframe(recommended_movies)
