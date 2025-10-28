# recommenders/kb.py
import pandas as pd
import numpy as np

def kb_recommend(user_id, music, rating_complete, preferred_genres=None, top_n=10):
    """
    Knowledge-Based Music Recommendation System
    --------------------------------------------
    Uses user's rating history and content metadata (genres, artists, score, etc.)
    to recommend songs that match their preferences.

    Parameters:
    -----------
    user_id : int
        The user ID for which recommendations are generated.
    music : pd.DataFrame
        Song metadata from music.csv or music_with_lyrics.
    rating_complete : pd.DataFrame
        User-song ratings dataframe with columns ['user_id', 'song_id', 'rating'].
    preferred_genres : list[str], optional
        User-selected genres for boosting recommendations.
    top_n : int
        Number of recommendations to return (default = 10).
    """

    if music.empty or 'Genres' not in music.columns:
        return pd.DataFrame({'Message': ['No music data available']})

    user_id = int(user_id)
    user_ratings = rating_complete[rating_complete['user_id'] == user_id]

    # === Step 1: Get user’s listened songs ===
    user_listened_ids = set(user_ratings['song_id'].values)

    # === Step 2: Analyze user preferences ===
    if not user_ratings.empty:
        # Merge with song metadata
        user_history = user_ratings.merge(
            music[['SID', 'Title', 'Artists', 'Genres', 'Score']],
            left_on='song_id',
            right_on='SID',
            how='left'
        )

        # Get highly rated songs (>= 7)
        high_rated = user_history[user_history['rating'] >= 7]

        # Genre preference frequency
        genre_freq = {}
        for genres in high_rated['Genres'].dropna():
            for g in str(genres).split(','):
                g = g.strip()
                genre_freq[g] = genre_freq.get(g, 0) + 1

        # Artist preference frequency
        artist_freq = {}
        for artist in high_rated['Artists'].dropna():
            for a in str(artist).split(','):
                a = a.strip()
                artist_freq[a] = artist_freq.get(a, 0) + 1

        user_avg_rating = user_ratings['rating'].mean()
    else:
        # No rating history
        genre_freq = {}
        artist_freq = {}
        user_avg_rating = 7.0

    # === Step 3: Merge user-selected genres if any ===
    if preferred_genres and isinstance(preferred_genres, (list, tuple, set)) and len(preferred_genres) > 0:
        for genre in preferred_genres:
            genre_freq[genre] = genre_freq.get(genre, 0) + 5  # boost selected genres

    # === Step 4: Filter out already listened songs ===
    candidate_songs = music[~music['SID'].isin(user_listened_ids)].copy()
    if candidate_songs.empty:
        return pd.DataFrame({'Message': ['User has listened to all available songs']})

    # === Step 5: Define scoring function ===
    def calculate_kb_score(row):
        score = 0

        # Base song quality (Score is 0-10 scale)
        if pd.notna(row['Score']):
            score += row['Score'] * 2  # weight: 2x

        # Genre match boost (up to +20)
        if pd.notna(row['Genres']) and genre_freq:
            genres = [g.strip() for g in str(row['Genres']).split(',')]
            genre_score = sum(genre_freq.get(g, 0) for g in genres)
            score += min(genre_score * 2, 20)

        # Artist preference boost (up to +10)
        if pd.notna(row['Artists']) and artist_freq:
            artists = [a.strip() for a in str(row['Artists']).split(',')]
            artist_score = sum(artist_freq.get(a, 0) for a in artists)
            score += min(artist_score * 2, 10)

        # Boost high-rated, popular songs (Score ≥ 8)
        if pd.notna(row['Score']) and row['Score'] >= 8.0:
            score += 5

        # Audio feature-based quality adjustments
        if 'danceability' in row and pd.notna(row['danceability']):
            score += row['danceability'] * 2
        if 'energy' in row and pd.notna(row['energy']):
            score += row['energy'] * 1.5
        if 'valence' in row and pd.notna(row['valence']):
            score += row['valence'] * 1.5

        return score

    # === Step 6: Calculate and rank ===
    candidate_songs['kb_score'] = candidate_songs.apply(calculate_kb_score, axis=1)
    top_recommendations = candidate_songs.nlargest(top_n, 'kb_score')

    # === Step 7: Return results ===
    result = top_recommendations[['SID', 'Title', 'Artists', 'Genres', 'Score']].copy()
    if 'lyrics' in top_recommendations.columns:
        result['lyrics'] = top_recommendations['lyrics']

    return result.reset_index(drop=True)
