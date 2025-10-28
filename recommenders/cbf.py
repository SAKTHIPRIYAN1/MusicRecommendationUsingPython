# recommenders/cbf.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

@st.cache_resource
def build_tfidf_matrix(music_with_lyrics: pd.DataFrame):
    """
    Build and cache a TF-IDF matrix for song lyrics.
    Returns:
        tfidf_matrix: sparse matrix of lyric features
        sid_to_pos: mapping of song IDs to matrix positions
    """
    tfidf = TfidfVectorizer(stop_words='english', max_features=7000)
    tfidf_matrix = tfidf.fit_transform(music_with_lyrics['lyrics'].fillna(''))
    sid_to_pos = {sid: pos for pos, sid in enumerate(music_with_lyrics['SID'])}
    return tfidf_matrix, sid_to_pos


def cbf_recommend(user_id: int, music_with_lyrics: pd.DataFrame, rating_complete: pd.DataFrame, top_n: int = 10):
    """
    Recommend songs to a user using content-based filtering (lyrics similarity).
    
    Args:
        user_id (int): user ID to recommend for
        music_with_lyrics (pd.DataFrame): songs with lyrics and metadata
        rating_complete (pd.DataFrame): user-song rating data
        top_n (int): number of recommendations to return
        
    Returns:
        DataFrame: top recommended songs with key info
    """
    # Songs listened by the user
    user_listened = rating_complete[rating_complete['user_id'] == int(user_id)]['song_id']
    music_with_lyrics = music_with_lyrics.reset_index(drop=True)

    # Get cached TF-IDF representation
    tfidf_matrix, sid_to_pos = build_tfidf_matrix(music_with_lyrics)

    # Map listened songs to TF-IDF positions
    listened_pos = [sid_to_pos[sid] for sid in user_listened if sid in sid_to_pos]

    if len(listened_pos) == 0:
        # No listening history — recommend top popular songs
        return music_with_lyrics.nlargest(top_n, 'Score')[['Title', 'Artists', 'Genres', 'Score', 'lyrics']]

    # Build user lyric profile
    user_profile = tfidf_matrix[listened_pos].mean(axis=0)
    user_profile_arr = np.asarray(user_profile)

    # Compute cosine similarity between user profile and all songs
    scores = cosine_similarity(user_profile_arr, tfidf_matrix).flatten()

    # Filter out already listened songs
    unlistened_pos = [i for i, sid in enumerate(music_with_lyrics['SID'])
                      if sid not in set(user_listened)]
    unlistened_scores = pd.Series(scores[unlistened_pos], index=unlistened_pos)

    # Get top recommendations
    top_pos = unlistened_scores.nlargest(top_n).index

    # Return clean DataFrame
    recs = music_with_lyrics.loc[top_pos][['Title', 'Artists', 'Genres', 'Score', 'lyrics']].copy()
    recs['similarity_score'] = unlistened_scores[top_pos].values
    return recs.sort_values(by='similarity_score', ascending=False)
