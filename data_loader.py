import pandas as pd
import os
import numpy as np


def extract_season_from_release_date(date_str):
    """
    Extract season from the 'release_date' column.
    Returns format: 'Season Year' (e.g., 'Spring 2019')
    """
    if pd.isna(date_str) or date_str == 'Unknown' or str(date_str).strip() == '':
        return 'Unknown'

    try:
        date_obj = pd.to_datetime(str(date_str), errors='coerce')
        if pd.isna(date_obj):
            return 'Unknown'
        year = date_obj.year
        month = date_obj.month
    except Exception:
        return 'Unknown'

    # Determine season
    if month in [1, 2, 3]:
        season = 'Winter'
    elif month in [4, 5, 6]:
        season = 'Spring'
    elif month in [7, 8, 9]:
        season = 'Summer'
    else:
        season = 'Fall'

    return f'{season} {year}'


def load_data():
    """
    Loads and cleans all music datasets for the recommendation system.
    Works in both Streamlit app and interactive mode.
    """
    # ✅ Resolve base path safely (even if __file__ is undefined)
    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = os.getcwd()

    base_path = os.path.join(base_dir, 'dataset')

    # ✅ File paths
    music_path = os.path.join(base_path, 'music.csv')
    lyrics_path = os.path.join(base_path, 'music_with_lyrics.csv')
    rating_path = os.path.join(base_path, 'rating_complete.csv')

    # ✅ Load datasets
    try:
        music = pd.read_csv(music_path)
        music_lyrics = pd.read_csv(lyrics_path)
        rating_complete = pd.read_csv(rating_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"❌ Missing one of the dataset files in: {base_path}\n{e}")

    # ✅ Validate required columns
    if 'SID' not in music.columns:
        raise KeyError("❌ 'SID' column not found in music.csv")
    if 'SID' not in music_lyrics.columns:
        raise KeyError("❌ 'SID' column not found in music_with_lyrics.csv")

    # ✅ Keep only songs that exist in lyrics file
    valid_ids = set(music_lyrics['SID'])
    music = music[music['SID'].isin(valid_ids)].copy()



    # ✅ Numeric column cleanup
    numeric_cols = [
        'Score', 'duration_ms', 'danceability', 'energy', 'valence',
        'acousticness', 'instrumentalness', 'tempo', 'speechiness',
        'liveness', 'loudness', 'Year'
    ]
    for col in numeric_cols:
        if col in music.columns:
            music[col] = pd.to_numeric(music[col], errors='coerce')

    if 'Score' in music.columns:
        music['Score'] = music['Score'].fillna(music['Score'].median())

    music = music.dropna(subset=['Score', 'duration_ms'])

    # ✅ Keep consistent SIDs across files
    valid_sids = set(music['SID'])
    music_lyrics = music_lyrics[music_lyrics['SID'].isin(valid_sids)].copy()
    if 'song_id' in rating_complete.columns:
        rating_complete = rating_complete[rating_complete['song_id'].isin(valid_sids)].copy()

    # ✅ Merge metadata and lyrics
    if 'lyrics' in music_lyrics.columns:
        music = music.merge(music_lyrics[['SID', 'lyrics']], on='SID', how='left')
    else:
        music['lyrics'] = 'No lyrics available'

    # ✅ Add Season from release_date
    if 'release_date' in music.columns:
        music['Season'] = music['release_date'].apply(extract_season_from_release_date)
    else:
        music['Season'] = 'Unknown'

    # ✅ Clean Genre column
    if 'Genres' not in music.columns:
        # Some datasets may call it "Genre"
        if 'Genre' in music.columns:
            music.rename(columns={'Genre': 'Genres'}, inplace=True)
        else:
            music['Genres'] = 'Unknown'

    # Standardize genres (remove brackets, quotes, etc.)
    music['Genres'] = (
        music['Genres']
        .astype(str)
        .replace(r'[\[\]\'"]', '', regex=True)
        .replace(r'\s*,\s*', ', ', regex=True)
        .fillna('Unknown')
    )

    # ✅ Drop irrelevant columns (keep Genres!)
    for col in ['release_date']:
        if col in music.columns:
            music = music.drop(columns=[col])

    # ✅ Reorder columns logically
    cols = music.columns.tolist()
    if 'lyrics' in cols:
        cols.remove('lyrics')
    if 'Season' in cols:
        cols.remove('Season')

    if 'Year' in cols:
        idx = cols.index('Year') + 1
        cols.insert(idx, 'Season')
    else:
        cols.append('Season')

    cols.append('lyrics')
    music = music[cols]

    print(f"✅ Loaded {len(music)} songs with cleaned metadata, genres, and lyrics.")
    return music, rating_complete
