import pandas as pd
import numpy as np
import difflib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# styling
# Injecting CSS for animation, hover and transformation
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.markdown("""
    <style>
    h1 {
        font-family: 'Trebuchet MS', sans-serif;
        color: #ffffff;
        background-color: #6c63ff;
        padding: 15px;
        border-radius: 10px;
    }
    .movie-card {
        background-color: #f0f0f5;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .movie-card:hover {
        transform: scale(1.03);
        box-shadow: 4px 4px 12px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    /* Default font for all text */
    body, .css-1d391kg, .css-ffhzg2 {  /* Streamlit container classes may change */
        font-family: 'Comic Sans MS', cursive, sans-serif;
        font-size: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Load dataset
df = pd.read_csv('Movies Recommendation.csv')
df.fillna('', inplace=True)

# Combine features
df['combined_features'] = df['Movie_Genre'] + ' ' + df['Movie_Keywords'] + ' ' + df['Movie_Tagline'] + ' ' + df['Movie_Cast'] + ' ' + df['Movie_Director']

# Vectorize
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['combined_features'])

# Similarity Score
Similarity_Score = cosine_similarity(X)

st.markdown("<h1 style='text-align: center;'>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)
st.markdown('<div class="center-container"><p>Movie Title basis Movie Recommendation</p>', unsafe_allow_html=True)

movie_input = st.text_input("Enter your favorite movie:")
top_n = st.selectbox("How many suggestions do you want?", [0,5, 10, 15, 20], index=1)

if st.button("Get Recommendations"):
    list_of_all_titles = df['Movie_Title'].tolist()
    Find_Close_Match = difflib.get_close_matches(movie_input, list_of_all_titles)

    if Find_Close_Match:
        Close_Match = Find_Close_Match[0]
        Index_of_Movie = df[df.Movie_Title == Close_Match]['Movie_ID'].values[0]
        Recommendation_Score = list(enumerate(Similarity_Score[Index_of_Movie]))
        sorted_similar_movies = sorted(Recommendation_Score, key=lambda x: x[1], reverse=True)[1:top_n+1]

        st.success(f"Top {top_n} movies similar to '{Close_Match}' are:")

        for i, movie in enumerate(sorted_similar_movies, 1):
            index = movie[0]
            selected_movie = df[df.Movie_ID == index].iloc[0]

            st.markdown(f"""
            <div style='border:1px solid #ccc;padding:10px;border-radius:10px;margin:10px 0;background-color:#f9f9f9;'>
                <h4>{i}. {selected_movie['Movie_Title']}</h4>
                <p><b>Genre:</b> {selected_movie['Movie_Genre']}</p>
                <p><b>Tagline:</b> {selected_movie['Movie_Tagline']}</p>
                <p><b>Cast:</b> {selected_movie['Movie_Cast']}</p>
                <p><b>Director:</b> {selected_movie['Movie_Director']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Movie not found. Please check the title.")



#==================================Genre=============

# Get unique genres
genres = sorted(df['Movie_Genre'].dropna().unique().tolist())

# CSS for centered rounded container & input styling + animations
st.markdown("""
<style>
.center-container {
    background: #121212;
    padding: 20px;
    max-width: 400px;
    margin: 30px auto;
    color:white;
    font-size:35px;
    font-family:cursive;
    border-radius: 10px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.6);
    transition: box-shadow 0.3s ease;
}
.center-container:hover {
    box-shadow: 0 12px 30px rgba(255, 230, 0, 0.8);
}
input, select {
    background: #222244;
    border-radius: 15px;
    padding: 12px 20px;
    margin-top: 15px;
    width: 100%;
    border: none;
    color: #fff;
    font-size: 18px;
    transition: box-shadow 0.3s ease;
}
input:focus, select:focus {
    outline: none;
    box-shadow: 0 0 10px #ffe600;
}
button {
    background: #ffe600;
    color: #121212;
    font-weight: bold;
    padding: 12px 25px;
    margin-top: 25px;
    border-radius: 20px;
    border: none;
    cursor: pointer;
    font-size: 18px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 15px #ffe600;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="center-container"><p>Gnere by Movie Recommendation</p>', unsafe_allow_html=True)

# Inputs centered in container
selected_genre = st.selectbox("Select Genre", genres)
num_recommend = st.select_slider("Number of Recommendations", options=[5, 10, 15, 20])
suggest_clicked = st.button("Suggest Movies")

st.markdown('</div>', unsafe_allow_html=True)

if suggest_clicked:
    filtered = df[df['Movie_Genre'].str.contains(selected_genre, case=False, na=False)]
    top_movies = filtered.sort_values('Movie_Vote', ascending=False).head(num_recommend)

    # Display movie cards with animation
    st.markdown("""
    <style>
    .movie-card {
        background-color: #1e1e2f;
        padding: 20px;
        margin: 10px 0;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, opacity 0.4s ease;
        opacity: 0.9;
    }
    .movie-card:hover {
        transform: scale(1.05);
        opacity: 1.0;
        background-color: #292945;
    }
    .movie-title {
        font-size: 22px;
        font-weight: bold;
        color: #ffe600;
    }
    .movie-info {
        font-size: 16px;
        color: #ddd;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader(f"Top {num_recommend} movies in genre '{selected_genre}'")

    for i, row in top_movies.iterrows():
        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-title">{row['Movie_Title']}</div>
            <div class="movie-info">🎭 Genre: {row['Movie_Genre']}</div>
            <div class="movie-info">👨‍🎤 Cast: {row['Movie_Cast']}</div>
            <div class="movie-info">🎬 Director: {row['Movie_Director']}</div>
            <div class="movie-info">🗒️ Tagline: {row['Movie_Tagline']}</div>
            <div class="movie-info">⭐ Rating: {row['Movie_Vote']}</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.write("Select genre and click 'Suggest Movies' to see recommendations.")


#=========================================================================Chart adn Graph=========================
import plotly.express as px

st.markdown('<div class="center-container"><p>Movie Budget vs Popularity based on genre</p>', unsafe_allow_html=True)

# Load data

# CSS and container code same as before (you can reuse the previous CSS + container code here)

selected_genre = st.selectbox("Select Genre2", genres)
num_recommend = st.select_slider(
    "Number of Recommendations",
    options=[5, 10, 15, 20],
        key="num_recommend_slider"
)
suggest_clicked = st.button("Suggest Movies_")

if suggest_clicked:
    # Filter movies by genre
    filtered = df[df['Movie_Genre'].str.contains(selected_genre, case=False, na=False)]
    
    # Top recommended movies based on vote/rating
    top_movies = filtered.sort_values('Movie_Vote', ascending=False).head(num_recommend)
    
    st.subheader(f"Top {num_recommend} movies in genre '{selected_genre}'")
    
    # Show movie info cards (reuse your previous movie card display code here)
    
    # Plotly charts for Budget & Popularity
    # Make sure your dataset has columns named 'Movie_Budget' and 'Movie_Popularity'
    # Convert to numeric (if needed)
    top_movies['Movie_Budget'] = pd.to_numeric(top_movies['Movie_Budget'], errors='coerce')
    top_movies['Movie_Popularity'] = pd.to_numeric(top_movies['Movie_Popularity'], errors='coerce')

    st.subheader("Movie Budget vs Popularity")

    # Scatter plot: Budget vs Popularity
    fig = px.scatter(
        top_movies,
        x='Movie_Budget',
        y='Movie_Popularity',
        text='Movie_Title',
        size='Movie_Vote',  # size by rating
        color='Movie_Vote',
        color_continuous_scale='Viridis',
        labels={"Movie_Budget": "Budget (in millions)", "Movie_Popularity": "Popularity"},
        title="Budget vs Popularity of Recommended Movies"
    )
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.write("Select genre and click 'Suggest Movies' to see recommendations and charts.")

    
