# main.py
import streamlit as st
import pandas as pd
from data_loader import load_data
from analytics import show_analytics
from recommenders.cbf import cbf_recommend
from recommenders.cf import cf_recommend
from recommenders.kb import kb_recommend
from recommenders.hybrid import hybrid_recommend

# ------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------
st.set_page_config(page_title="Music Recommendation System", layout="wide")

st.title("Music Recommendation System")

# Load data
music, rating_complete = load_data()

st.sidebar.header(" User Input & Filters")
user_id = st.sidebar.text_input("Enter User ID", "1")

# ------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------

# Genre filter
genre_options = sorted(set(g.strip() for gs in music['Genres'].dropna().str.split(',') for g in gs))
selected_genres = st.sidebar.multiselect("Filter by Genres", genre_options)

# Year filter
min_year, max_year = int(music['Year'].min()), int(music['Year'].max())
year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))

# Energy / Danceability sliders
danceability_range = st.sidebar.slider(
    "Danceability",
    float(music['danceability'].min()),
    float(music['danceability'].max()),
    (float(music['danceability'].min()), float(music['danceability'].max()))
)
energy_range = st.sidebar.slider(
    "Energy",
    float(music['energy'].min()),
    float(music['energy'].max()),
    (float(music['energy'].min()), float(music['energy'].max()))
)

# Explicit filter
explicit_filter = st.sidebar.selectbox("Explicit Content", ["Any", "Explicit", "Non-explicit"])

# ------------------------------------------------------
# FILTER MUSIC DATASET BASED ON USER SELECTION
# ------------------------------------------------------
filtered_music = music.copy()

# Genre filter
if selected_genres:
    filtered_music = filtered_music[
        filtered_music['Genres'].apply(
            lambda x: any(g in x for g in selected_genres) if pd.notnull(x) else False
        )
    ]

# Apply filters
filtered_music = filtered_music[
    (filtered_music['Year'].between(year_range[0], year_range[1])) &
    (filtered_music['danceability'].between(danceability_range[0], danceability_range[1])) &
    (filtered_music['energy'].between(energy_range[0], energy_range[1]))
]

# Explicit
if explicit_filter != "Any":
    if explicit_filter == "Explicit":
        filtered_music = filtered_music[filtered_music['explicit'] == 1]
    else:
        filtered_music = filtered_music[filtered_music['explicit'] == 0]

# Cleanup — remove blank rows and reindex
filtered_music = filtered_music.dropna(how="all").reset_index(drop=True)

if filtered_music.empty:
    st.warning("No songs match your selected filters.")

# ------------------------------------------------------
# STREAMLIT TABS
# ------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Analytics & Insights", " Recommendations", " Find Similar Songs", "User Analytics", "Global Trends"
])

# ------------------------------------------------------
# TAB 1: ANALYTICS
# ------------------------------------------------------
with tab1:
    st.header(" Analytics & Visualizations")
    show_analytics(music, rating_complete)

# ------------------------------------------------------
# TAB 2: RECOMMENDATIONS
# ------------------------------------------------------
with tab2:
    st.header(" Personalized Recommendations")
    st.caption("Recommendations are personalized based on your ratings and filters.")
    model_choice = st.radio(
        "Choose Recommendation Model",
        ["Content-Based", "Collaborative Filtering", "Knowledge-Based", "Hybrid"]
    )

    if model_choice == "Content-Based":
        st.subheader("Content-Based Recommendations")
        recs_cbf = cbf_recommend(user_id, filtered_music, rating_complete)
        if not recs_cbf.empty:
            st.dataframe(recs_cbf[['Title', 'Artists', 'Genres', 'lyrics']])
        else:
            st.warning("No recommendations found for this user.")

    elif model_choice == "Collaborative Filtering":
        st.subheader("Collaborative Filtering Recommendations")
        recs_cf = cf_recommend(user_id, rating_complete, filtered_music)
        if not recs_cf.empty and 'Title' in recs_cf.columns:
            st.dataframe(recs_cf[['Title', 'Artists', 'Genres', 'predicted_rating', 'lyrics']])
            st.subheader("Top Predicted Ratings")
            st.bar_chart(recs_cf.set_index('Title')['predicted_rating'].head(10))
        else:
            st.warning("No CF recommendations available for this user.")

    elif model_choice == "Knowledge-Based":
        st.subheader("Knowledge-Based Recommendations")
        recs_kb = kb_recommend(user_id, filtered_music, rating_complete, selected_genres)
        if not recs_kb.empty:
            st.dataframe(recs_kb[['Title', 'Artists', 'Genres', 'lyrics']])
        else:
            st.warning("No KB recommendations found for the selected genres.")

    elif model_choice == "Hybrid":
        st.subheader("Hybrid Recommendations (CBF + CF + KB)")
        recs_hybrid = hybrid_recommend(user_id, filtered_music, rating_complete, selected_genres=selected_genres)
        if not recs_hybrid.empty:
            st.dataframe(recs_hybrid[['Title', 'Artists', 'Genres', 'hybrid_score', 'reasoning', 'lyrics']])
        else:
            st.warning("No hybrid recommendations found.")

# ------------------------------------------------------
# TAB 3: FIND SIMILAR SONGS
# ------------------------------------------------------
with tab3:
    st.header(" Find Similar Songs")
    song_names = list(music['Title'].dropna().unique())
    selected_song = st.selectbox("Select a Song", ["None"] + song_names)
    genre_weight = st.slider("Genre Similarity Weight", 0.0, 2.0, 1.0, 0.1)

    if selected_song != "None":
        selected_row = music[music['Title'] == selected_song]
        if not selected_row.empty:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(music['lyrics'].fillna(''))
            idx = selected_row.index[0]
            sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            genre_sim = music['Genres'].apply(
                lambda x: len(set(str(x).split(",")) & set(str(selected_row.iloc[0]['Genres']).split(","))) if pd.notnull(x) else 0
            )
            combined_score = sim_scores + genre_weight * genre_sim
            top_idx = pd.Series(combined_score).nlargest(11).index
            similar_songs = music.iloc[top_idx][['Title', 'Artists', 'Genres', 'lyrics']]
            similar_songs = similar_songs[similar_songs['Title'] != selected_song].head(10)
            st.write(similar_songs)

# ------------------------------------------------------
# TAB 4: USER ANALYTICS
# ------------------------------------------------------
with tab4:
    st.header(" User Analytics")
    user_id_int = int(user_id)
    user_ratings = rating_complete[rating_complete['user_id'] == user_id_int]

    if not user_ratings.empty:
        user_history = user_ratings.merge(music, left_on='song_id', right_on='SID', how='left')

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("User Rating Distribution")
            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, ax = plt.subplots()
            sns.histplot(user_ratings['rating'], bins=10, ax=ax, kde=True)
            ax.set_xlabel('Rating')
            ax.set_ylabel('Count')
            st.pyplot(fig)

        with col2:
            st.subheader("Genre Preferences")
            genre_list = []
            for genres in user_history['Genres'].dropna():
                genre_list.extend([g.strip() for g in str(genres).split(',')])
            if genre_list:
                genre_counts = pd.Series(genre_list).value_counts()
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%', startangle=90)
                ax.set_title('Preferred Music Genres')
                st.pyplot(fig)
            else:
                st.info("No genre data available for this user.")

        st.subheader("Recent Ratings")
        history_display = user_history[['Title', 'Artists', 'rating', 'Genres']].sort_values('rating', ascending=False)
        st.dataframe(history_display)

        st.subheader("User Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Songs Rated", len(user_ratings))
        col2.metric("Average Rating", f"{user_ratings['rating'].mean():.2f}")
        col3.metric("Top Genre", genre_counts.index[0] if len(genre_list) > 0 else "N/A")

    else:
        st.warning("No ratings found for this user.")

# ------------------------------------------------------
# TAB 5: GLOBAL MUSIC INSIGHTS
# ------------------------------------------------------
with tab5:
    st.header(" Global Music Trends")
    st.caption("Explore globally trending or top-rated songs (not user-specific).")

    global_filtered_music = music.copy()
    if selected_genres:
        global_filtered_music = global_filtered_music[
            global_filtered_music['Genres'].apply(
                lambda x: any(g in x for g in selected_genres) if pd.notnull(x) else False
            )
        ]
    global_filtered_music = global_filtered_music[
        (global_filtered_music['Year'].between(year_range[0], year_range[1]))
    ]

    st.write(" Top Rated Songs")
    st.dataframe(global_filtered_music.sort_values("Year", ascending=False).head(10)[["Title", "Artists", "Genres"]])

    st.write(" Most Recent Hits")
    st.dataframe(global_filtered_music.sort_values("Year", ascending=False).head(10)[["Title", "Artists", "Genres", "Year"]])
