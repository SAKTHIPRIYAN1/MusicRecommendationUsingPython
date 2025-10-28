# analytics.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def show_analytics(music, rating_complete):
    st.title(" Music Dataset Analytics")

    # ====== Music Score / Popularity Distribution ======
    st.subheader("Song Popularity (Score) Distribution")
    if 'Score' in music.columns:
        fig, ax = plt.subplots()
        sns.histplot(music['Score'].dropna(), bins=20, ax=ax, color='skyblue')
        ax.set_xlabel("Score / Popularity")
        ax.set_ylabel("Number of Songs")
        st.pyplot(fig)
    else:
        st.info("No 'Score' column found in dataset.")

    # ======  Top Artists ======
    st.subheader("Top 10 Most Frequent Artists")
    if 'Artists' in music.columns:
        artists = music['Artists'].dropna().str.split(',').explode().str.strip()
        top_artists = artists.value_counts().head(10)
        st.bar_chart(top_artists)
    else:
        st.info("No 'Artists' column found in dataset.")

    # ====== 3️ Top Genres (if available) ======
    if 'Genres' in music.columns:
        st.subheader("Top Genres")
        genres = music['Genres'].dropna().str.split(',').explode().str.strip()
        top_genres = genres.value_counts().head(10)
        st.bar_chart(top_genres)

    # ====== 4️ Energy vs Danceability Scatter ======
    st.subheader("Energy vs Danceability")
    if all(col in music.columns for col in ['energy', 'danceability']):
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=music, x='danceability', y='energy', alpha=0.6)
        ax2.set_xlabel("Danceability")
        ax2.set_ylabel("Energy")
        st.pyplot(fig2)
    else:
        st.info("Missing columns 'energy' or 'danceability' for scatter plot.")

    # ====== 5️ User Rating Distribution ======
    st.subheader("User Rating Distribution")
    if 'rating' in rating_complete.columns:
        fig3, ax3 = plt.subplots()
        sns.histplot(rating_complete['rating'].dropna(), bins=20, ax=ax3, color='salmon')
        ax3.set_xlabel("User Rating")
        ax3.set_ylabel("Count")
        st.pyplot(fig3)
    else:
        st.info("No 'rating' column found in user ratings dataset.")
