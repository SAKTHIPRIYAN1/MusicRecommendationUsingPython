import pandas as pd
import numpy as np
import os
from datetime import datetime
import random

# ========== CONFIG ==========
BASE_PATH = '/mnt/D/BDA/python/Anime Recommendation System/dataset'
INPUT_FILE = os.path.join(BASE_PATH, 'musicDataset.csv')

# ========== LOAD ==========
music = pd.read_csv(INPUT_FILE)
music.columns = music.columns.str.strip()

# ---------- 1️⃣ CREATE music.csv ----------
music_csv = music.copy()

# Keep essential columns
keep_cols = [
    'sid', 'name', 'artists', 'year', 'popularity', 'duration_ms',
    'danceability', 'energy', 'valence', 'acousticness',
    'instrumentalness', 'tempo', 'explicit', 'speechiness',
    'liveness', 'loudness', 'key', 'mode', 'release_date'
]
music_csv = music_csv[[col for col in keep_cols if col in music.columns]]

# ====== Generate Mood & Genre ======
def get_mood(row):
    if row['valence'] > 0.7 and row['energy'] > 0.6:
        return 'Happy'
    elif row['valence'] < 0.4 and row['acousticness'] > 0.5:
        return 'Calm'
    elif row['energy'] > 0.7 and row['valence'] < 0.5:
        return 'Energetic'
    else:
        return 'Neutral'

music_csv['Mood'] = music_csv.apply(get_mood, axis=1)

def infer_genre(row):
    t = row['tempo']
    if t < 70:
        return 'Classical'
    elif t < 100:
        return 'Jazz'
    elif t < 130:
        return 'Pop'
    else:
        return 'Rock'

music_csv['Genres'] = music_csv.apply(infer_genre, axis=1)

# 🎯 Ensure popularity >= 10 (make small ones more presentable)
music_csv.loc[music_csv['popularity'] < 10, 'popularity'] = [
    random.randint(10, 25) for _ in range(sum(music_csv['popularity'] < 10))
]

# Reorder columns
music_csv = music_csv[
    ['sid', 'name', 'artists', 'Genres', 'Mood', 'year', 'popularity',
     'duration_ms', 'danceability', 'energy', 'valence',
     'acousticness', 'instrumentalness', 'tempo', 'explicit',
     'speechiness', 'liveness', 'loudness', 'key', 'mode', 'release_date']
]

music_csv.rename(columns={
    'sid': 'SID',
    'name': 'Title',
    'artists': 'Artists',
    'year': 'Year',
    'popularity': 'Score'
}, inplace=True)

output_csv = os.path.join(BASE_PATH, 'music.csv')
music_csv.to_csv(output_csv, index=False)
print(f"✅ music.csv created at {output_csv}")

# ---------- 2️⃣ CREATE music_with_lyrics.csv ----------
music_with_lyrics = music_csv[['SID', 'Title', 'Artists', 'Genres', 'Mood', 'Score']].copy()

# 💬 Generate descriptive text dynamically
def generate_description(row):
    energy_level = (
        "high energy" if row['Mood'] in ['Energetic', 'Happy']
        else "soft and calm" if row['Mood'] == 'Calm'
        else "balanced and smooth"
    )
    tempo_desc = (
        "slow rhythm" if row['Genres'] == 'Classical'
        else "mellow flow" if row['Genres'] == 'Jazz'
        else "catchy beat" if row['Genres'] == 'Pop'
        else "fast-paced vibe"
    )
    return (
        f"{row['Title']} is a {row['Mood'].lower()} {row['Genres'].lower()} track "
        f"by {row['Artists']}. Released in {row['Year']}, "
        f"it features {energy_level} with a {tempo_desc}, "
        f"and reflects a valence of {row['valence']:.2f} and energy of {row['energy']:.2f}."
    )

music_with_lyrics['lyrics'] = music_csv.apply(generate_description, axis=1)

output_lyrics = os.path.join(BASE_PATH, 'music_with_lyrics.csv')
music_with_lyrics.to_csv(output_lyrics, index=False)
print(f"✅ music_with_lyrics.csv created at {output_lyrics}")

# ---------- 3️⃣ CREATE rating_complete.csv ----------
unique_songs = music_csv['SID'].unique()
num_users = 500
ratings = []

for user_id in range(1, num_users + 1):
    rated_songs = np.random.choice(unique_songs, size=min(30, len(unique_songs)), replace=False)
    for sid in rated_songs:
        rating = random.randint(1, 10)
        ratings.append({'user_id': user_id, 'song_id': sid, 'rating': rating})

rating_df = pd.DataFrame(ratings)
output_ratings = os.path.join(BASE_PATH, 'rating_complete.csv')
rating_df.to_csv(output_ratings, index=False)
print(f"✅ rating_complete.csv created at {output_ratings}")

print("\n🎵 All 3 CSVs successfully generated in:", BASE_PATH)
