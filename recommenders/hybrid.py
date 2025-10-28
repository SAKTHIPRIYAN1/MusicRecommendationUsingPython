# recommenders/hybrid.py
import pandas as pd
import numpy as np
from recommenders.cbf import cbf_recommend
from recommenders.cf import cf_recommend
from recommenders.kb import kb_recommend

def hybrid_recommend(user_id, music, rating_complete, selected_genres=None, top_n=10):
    """
    Hybrid music recommendation combining CBF, CF, and KB methods.
    Each recommender contributes weighted normalized scores to the final ranking.
    
    Parameters:
        user_id (int): Target user ID for recommendation
        music (pd.DataFrame): Music metadata (from music.csv or music_with_lyrics.csv)
        rating_complete (pd.DataFrame): User-song ratings dataframe
        selected_genres (list[str]): Optional genres for KB recommendations
        top_n (int): Number of recommendations to return
    """
    # === Step 1: Run all recommenders ===
    cbf_recs = cbf_recommend(user_id, music, rating_complete, top_n=20)
    cf_recs = cf_recommend(user_id, rating_complete, music, top_n=20)
    kb_recs = kb_recommend(user_id, music, rating_complete, selected_genres, top_n=20)

    # === Step 2: Normalize scores to 0–10 scale ===
    def normalize_scores(df, score_col, name_col='Title'):
        if df is None or df.empty or score_col not in df.columns:
            return {}
        scores = {}
        max_score = df[score_col].max()
        min_score = df[score_col].min()
        if max_score == min_score:
            for _, row in df.iterrows():
                scores[row[name_col]] = 10.0
        else:
            for _, row in df.iterrows():
                normalized = 10 * (row[score_col] - min_score) / (max_score - min_score)
                scores[row[name_col]] = normalized
        return scores

    cbf_scores = normalize_scores(cbf_recs, 'Score')
    kb_scores = normalize_scores(kb_recs, 'Score')

    # CF scores are already in 1–10 range
    cf_scores = {}
    if cf_recs is not None and not cf_recs.empty and 'Title' in cf_recs.columns:
        for _, row in cf_recs.iterrows():
            if pd.notna(row['Title']):
                cf_scores[row['Title']] = row['predicted_rating']

    # === Step 3: Weighted Combination ===
    hybrid_data = {}
    all_songs = set(list(cbf_scores.keys()) + list(cf_scores.keys()) + list(kb_scores.keys()))

    for title in all_songs:
        cbf_score = cbf_scores.get(title, 0)
        cf_score = cf_scores.get(title, 0)
        kb_score = kb_scores.get(title, 0)

        total_weight = 0
        weighted_sum = 0
        methods_used = []

        if cbf_score > 0:
            weighted_sum += cbf_score * 0.45
            total_weight += 0.45
            methods_used.append("content similarity (audio + lyrics)")
        if cf_score > 0:
            weighted_sum += cf_score * 0.30
            total_weight += 0.30
            methods_used.append("similar users' preferences")
        if kb_score > 0:
            weighted_sum += kb_score * 0.25
            total_weight += 0.25
            methods_used.append("high-rated songs in your favorite genres")

        if total_weight > 0:
            final_score = weighted_sum / total_weight

            # Create reasoning explanation
            if len(methods_used) == 1:
                reasoning = f"Recommended based on {methods_used[0]}."
            elif len(methods_used) == 2:
                reasoning = f"Recommended based on {methods_used[0]} and {methods_used[1]}."
            else:
                reasoning = f"Strongly recommended based on {', '.join(methods_used[:-1])}, and {methods_used[-1]}."

            hybrid_data[title] = {
                'score': final_score,
                'reasoning': reasoning
            }

    # === Step 4: Sort by hybrid score ===
    sorted_songs = sorted(hybrid_data.items(), key=lambda x: x[1]['score'], reverse=True)[:top_n]

    # === Step 5: Build final DataFrame ===
    result_data = []
    for title, data in sorted_songs:
        song_row = music[music['Title'] == title]
        if not song_row.empty:
            result_data.append({
                'Title': title,
                'Artists': song_row.iloc[0]['Artists'],
                'Genres': song_row.iloc[0]['Genres'],
                'hybrid_score': round(data['score'], 2),
                'Score': song_row.iloc[0]['Score'],
                'reasoning': data['reasoning'],
                'lyrics': song_row.iloc[0].get('lyrics', None),
                'SID': song_row.iloc[0]['SID']
            })

    return pd.DataFrame(result_data)
