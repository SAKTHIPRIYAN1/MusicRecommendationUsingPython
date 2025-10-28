# recommenders/cf.py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import streamlit as st

@st.cache_resource
def build_cf_model(rating_complete):
    """Build and cache collaborative filtering model for music recommendations."""
    
    # Filter users with at least 10 ratings (for better quality)
    user_counts = rating_complete['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 10].index
    filtered = rating_complete[rating_complete['user_id'].isin(valid_users)]
    
    # Keep top 2500 songs for balance
    song_counts = filtered['song_id'].value_counts()
    top_songs = song_counts.head(2500).index
    filtered = filtered[filtered['song_id'].isin(top_songs)]
    
    # Create user and song mappings
    user_map = {uid: idx for idx, uid in enumerate(filtered['user_id'].unique())}
    song_map = {sid: idx for idx, sid in enumerate(filtered['song_id'].unique())}
    
    # Build sparse rating matrix
    row = filtered['user_id'].map(user_map)
    col = filtered['song_id'].map(song_map)
    data = filtered['rating']
    ratings_sparse = csr_matrix((data, (row, col)), shape=(len(user_map), len(song_map)))
    
    # Build item-based CF model
    item_model = NearestNeighbors(
        metric='cosine',
        algorithm='brute',
        n_neighbors=min(30, ratings_sparse.shape[1]),
        n_jobs=-1
    )
    item_model.fit(ratings_sparse.T)
    
    # Build user-based CF model
    user_model = NearestNeighbors(
        metric='cosine',
        algorithm='brute',
        n_neighbors=min(50, ratings_sparse.shape[0]),
        n_jobs=-1
    )
    user_model.fit(ratings_sparse)
    
    # Compute user and item means
    user_means = np.array(ratings_sparse.mean(axis=1)).flatten()
    item_means = np.array(ratings_sparse.mean(axis=0)).flatten()
    
    # Compute user standard deviations
    user_stds = np.zeros(len(user_means))
    for i in range(ratings_sparse.shape[0]):
        user_ratings = ratings_sparse[i].toarray().flatten()
        rated_mask = user_ratings > 0
        if rated_mask.sum() > 1:
            user_stds[i] = np.std(user_ratings[rated_mask])
        else:
            user_stds[i] = 1.0
    user_stds[user_stds == 0] = 1.0
    
    return ratings_sparse, user_map, song_map, user_model, item_model, user_means, item_means, user_stds


def cf_recommend(user_id, rating_complete, music_with_lyrics=None, top_n=10):
    """Collaborative filtering music recommender system."""
    try:
        user_id = int(user_id)
        top_n = int(top_n)
    except (ValueError, TypeError):
        return pd.DataFrame({'Message': ['Invalid user_id or top_n parameter']})
    
    # Load CF model
    ratings_sparse, user_map, song_map, user_model, item_model, user_means, item_means, user_stds = build_cf_model(rating_complete)
    
    if user_id not in user_map:
        return pd.DataFrame({'Message': ['User not found or too few ratings']})
    
    user_idx = user_map[user_id]
    user_ratings = ratings_sparse[user_idx].toarray().flatten()
    rated_indices = np.where(user_ratings > 0)[0]
    
    if len(rated_indices) == 0:
        return pd.DataFrame({'Message': ['User has not rated any songs']})
    
    user_mean = user_means[user_idx]
    user_std = user_stds[user_idx]
    
    # === USER-BASED CF PREDICTIONS ===
    user_based_predictions = np.full(ratings_sparse.shape[1], user_mean)
    
    distances, similar_users = user_model.kneighbors(
        ratings_sparse[user_idx], 
        n_neighbors=min(30, ratings_sparse.shape[0])
    )
    
    similar_users = similar_users.flatten()[1:]
    similarities = 1 - distances.flatten()[1:]
    
    valid_mask = similarities > 0.2
    similar_users = similar_users[valid_mask]
    similarities = similarities[valid_mask]
    
    if len(similar_users) > 0:
        similar_ratings = ratings_sparse[similar_users].toarray()
        similar_means = user_means[similar_users]
        
        for song_idx in range(ratings_sparse.shape[1]):
            if song_idx not in rated_indices:
                song_ratings = similar_ratings[:, song_idx]
                rated_mask = song_ratings > 0
                
                if rated_mask.sum() > 0:
                    centered = song_ratings[rated_mask] - similar_means[rated_mask]
                    weighted_sum = np.sum(centered * similarities[rated_mask])
                    sim_sum = np.sum(similarities[rated_mask])
                    user_based_predictions[song_idx] = user_mean + (weighted_sum / sim_sum)
    
    # === ITEM-BASED CF PREDICTIONS ===
    item_based_predictions = np.full(ratings_sparse.shape[1], user_mean)
    item_weights = np.zeros(ratings_sparse.shape[1])
    
    top_rated_items = sorted(rated_indices, key=lambda x: user_ratings[x], reverse=True)[:20]
    
    for rated_idx in top_rated_items:
        distances, similar_items = item_model.kneighbors(
            ratings_sparse.T[rated_idx],
            n_neighbors=min(20, ratings_sparse.shape[1])
        )
        
        similar_items = similar_items.flatten()[1:]
        similarities = 1 - distances.flatten()[1:]
        
        valid_mask = (similarities > 0.15) & (~np.isin(similar_items, rated_indices))
        similar_items = similar_items[valid_mask]
        similarities = similarities[valid_mask]
        
        if len(similar_items) > 0:
            normalized_rating = (user_ratings[rated_idx] - user_mean) / user_std
            for sim_idx, sim in zip(similar_items, similarities):
                item_based_predictions[sim_idx] += normalized_rating * sim
                item_weights[sim_idx] += sim
    
    # Normalize item-based predictions
    item_weights[item_weights == 0] = 1
    item_based_predictions = user_mean + (item_based_predictions - user_mean) * user_std / item_weights
    
    # === COMBINE BOTH MODELS ===
    final_predictions = np.full(ratings_sparse.shape[1], user_mean)
    user_valid = np.abs(user_based_predictions - user_mean) > 0.1
    item_valid = item_weights > 0
    
    both = user_valid & item_valid
    user_only = user_valid & ~item_valid
    item_only = ~user_valid & item_valid
    
    final_predictions[both] = (
        0.6 * user_based_predictions[both] + 0.4 * item_based_predictions[both]
    )
    final_predictions[user_only] = user_based_predictions[user_only]
    final_predictions[item_only] = item_based_predictions[item_only]
    
    final_predictions = np.clip(final_predictions, 1, 10)
    final_predictions[rated_indices] = -np.inf  # ignore already rated songs
    
    top_indices = np.argsort(-final_predictions)[:top_n * 2]
    song_inv_map = {v: k for k, v in song_map.items()}
    
    rec_data = []
    for idx in top_indices:
        if final_predictions[idx] > user_mean - 1.0 and len(rec_data) < top_n:
            song_id = song_inv_map[idx]
            score = float(final_predictions[idx])
            rec_data.append({
                'song_id': song_id,
                'predicted_rating': round(score, 2)
            })
    
    # Merge with song data if provided
    result_df = pd.DataFrame(rec_data)
    
    if music_with_lyrics is not None and not result_df.empty:
        result_df = result_df.merge(
            music_with_lyrics[['SID', 'Title', 'Artists', 'Genres', 'Score', 'lyrics']],
            left_on='song_id',
            right_on='SID',
            how='left'
        )
    
    return result_df
