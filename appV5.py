# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # from numpy import load
# # # from scipy.sparse import load_npz
# # # from joblib import load as joblib_load
# # # from content_based_filtering import content_recommendation
# # # from collaborative_filtering import collaborative_recommendation
# # # from hybrid_recommendations import HybridRecommenderSystem
# # # from sklearn.metrics.pairwise import cosine_similarity

# # # # --------- STREAMLIT-COMPATIBLE CSS ---------
# # # st.markdown("""
# # # <style>
# # #     /* Hide Streamlit branding */
# # #     #MainMenu {visibility: hidden;}
# # #     footer {visibility: hidden;}
# # #     header {visibility: hidden;}
    
# # #     /* Main app background */
# # #     .stApp {
# # #         background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
# # #     }
    
# # #     /* Main container styling */
# # #     .main .block-container {
# # #         padding-top: 1rem;
# # #         padding-bottom: 2rem;
# # #         max-width: 1200px;
# # #     }
    
# # #     /* Header styling */
# # #     .main-title {
# # #         font-size: 3rem !important;
# # #         font-weight: bold !important;
# # #         text-align: center !important;
# # #         background: linear-gradient(90deg, #1DB954, #1ed760) !important;
# # #         -webkit-background-clip: text !important;
# # #         -webkit-text-fill-color: transparent !important;
# # #         background-clip: text !important;
# # #         margin-bottom: 0.5rem !important;
# # #         padding: 1rem 0 !important;
# # #     }
    
# # #     .subtitle {
# # #         text-align: center !important;
# # #         font-size: 1.2rem !important;
# # #         color: #666 !important;
# # #         margin-bottom: 2rem !important;
# # #     }
    
# # #     /* Sidebar improvements */
# # #     .css-1d391kg {
# # #         background: linear-gradient(180deg, #1DB954 0%, #1a8f47 100%) !important;
# # #     }
    
# # #     .sidebar .sidebar-content {
# # #         background: transparent !important;
# # #     }
    
# # #     /* Metric cards */
# # #     .metric-container {
# # #         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
# # #         padding: 1.5rem !important;
# # #         border-radius: 10px !important;
# # #         color: white !important;
# # #         text-align: center !important;
# # #         margin: 0.5rem 0 !important;
# # #         box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
# # #     }
    
# # #     .metric-value {
# # #         font-size: 2rem !important;
# # #         font-weight: bold !important;
# # #         margin-bottom: 0.5rem !important;
# # #     }
    
# # #     .metric-label {
# # #         font-size: 0.9rem !important;
# # #         opacity: 0.9 !important;
# # #     }
    
# # #     /* Song card styling */
# # #     .song-card {
# # #         background: white !important;
# # #         padding: 1.5rem !important;
# # #         border-radius: 10px !important;
# # #         margin: 1rem 0 !important;
# # #         border-left: 4px solid #1DB954 !important;
# # #         box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
# # #         transition: transform 0.2s ease !important;
# # #     }
    
# # #     .song-card:hover {
# # #         transform: translateY(-2px) !important;
# # #         box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
# # #     }
    
# # #     .song-title {
# # #         font-size: 1.3rem !important;
# # #         font-weight: bold !important;
# # #         color: #2d3748 !important;
# # #         margin-bottom: 0.3rem !important;
# # #     }
    
# # #     .song-artist {
# # #         font-size: 1rem !important;
# # #         color: #4a5568 !important;
# # #         margin-bottom: 0.5rem !important;
# # #     }
    
# # #     .song-match {
# # #         font-size: 0.9rem !important;
# # #         color: #1DB954 !important;
# # #         font-weight: bold !important;
# # #     }
    
# # #     /* Button improvements */
# # #     .stButton > button {
# # #         background: linear-gradient(135deg, #1DB954, #1ed760) !important;
# # #         color: white !important;
# # #         border: none !important;
# # #         border-radius: 20px !important;
# # #         padding: 0.5rem 1rem !important;
# # #         font-weight: bold !important;
# # #         transition: all 0.2s ease !important;
# # #     }
    
# # #     .stButton > button:hover {
# # #         background: linear-gradient(135deg, #1ed760, #22c55e) !important;
# # #         transform: translateY(-1px) !important;
# # #     }
    
# # #     /* Status badges */
# # #     .status-badge {
# # #         display: inline-block !important;
# # #         padding: 0.25rem 0.75rem !important;
# # #         margin: 0.25rem !important;
# # #         border-radius: 15px !important;
# # #         font-size: 0.8rem !important;
# # #         font-weight: bold !important;
# # #     }
    
# # #     .badge-liked {
# # #         background: #fee2e2 !important;
# # #         color: #dc2626 !important;
# # #     }
    
# # #     .badge-disliked {
# # #         background: #dbeafe !important;
# # #         color: #2563eb !important;
# # #     }
    
# # #     .badge-queued {
# # #         background: #f3e8ff !important;
# # #         color: #7c3aed !important;
# # #     }
    
# # #     /* Playing indicators */
# # #     .now-playing {
# # #         background: #1DB954 !important;
# # #         color: white !important;
# # #         padding: 0.5rem 1rem !important;
# # #         border-radius: 20px !important;
# # #         font-size: 0.9rem !important;
# # #         font-weight: bold !important;
# # #         margin-bottom: 1rem !important;
# # #         display: inline-block !important;
# # #     }
    
# # #     .next-up {
# # #         background: #e5f3ff !important;
# # #         color: #1DB954 !important;
# # #         padding: 0.5rem 1rem !important;
# # #         border-radius: 20px !important;
# # #         font-size: 0.9rem !important;
# # #         font-weight: bold !important;
# # #         margin-bottom: 1rem !important;
# # #         display: inline-block !important;
# # #         border: 1px solid #1DB954 !important;
# # #     }
    
# # #     /* Audio preview styling */
# # #     .audio-section {
# # #         background: #f8fafc !important;
# # #         padding: 1rem !important;
# # #         border-radius: 8px !important;
# # #         margin-top: 1rem !important;
# # #         border: 1px solid #e2e8f0 !important;
# # #     }
    
# # #     /* Mood active indicator */
# # #     .mood-indicator {
# # #         background: #1DB954 !important;
# # #         color: white !important;
# # #         padding: 0.5rem 1rem !important;
# # #         border-radius: 20px !important;
# # #         font-weight: bold !important;
# # #         margin: 0.5rem 0 !important;
# # #         display: inline-block !important;
# # #     }
    
# # #     /* Form improvements */
# # #     .stSelectbox > div > div {
# # #         border-radius: 8px !important;
# # #     }
    
# # #     .stTextInput > div > div > input {
# # #         border-radius: 8px !important;
# # #     }
    
# # #     /* Success/Error message styling */
# # #     .stSuccess {
# # #         background: #f0fdf4 !important;
# # #         border: 1px solid #22c55e !important;
# # #         border-radius: 8px !important;
# # #     }
    
# # #     .stError {
# # #         background: #fef2f2 !important;
# # #         border: 1px solid #ef4444 !important;
# # #         border-radius: 8px !important;
# # #     }
    
# # #     .stWarning {
# # #         background: #fffbeb !important;
# # #         border: 1px solid #f59e0b !important;
# # #         border-radius: 8px !important;
# # #     }
# # # </style>
# # # """, unsafe_allow_html=True)

# # # # ------ SESSION STATE (unchanged) ------
# # # if 'actions' not in st.session_state:
# # #     st.session_state.actions = {}
# # # if 'playlists' not in st.session_state:
# # #     st.session_state.playlists = {}
# # # if 'recommendation_history' not in st.session_state:
# # #     st.session_state.recommendation_history = []
# # # if 'user_preferences' not in st.session_state:
# # #     st.session_state.user_preferences = {}
# # # if 'liked_songs' not in st.session_state:
# # #     st.session_state.liked_songs = set()
# # # if 'disliked_songs' not in st.session_state:
# # #     st.session_state.disliked_songs = set()
# # # if 'queued_songs' not in st.session_state:
# # #     st.session_state.queued_songs = set()
# # # if 'selected_mood' not in st.session_state:
# # #     st.session_state.selected_mood = "Custom"
# # # if 'current_recommendations' not in st.session_state:
# # #     st.session_state.current_recommendations = pd.DataFrame()
# # # if 'playlist_actions' not in st.session_state:
# # #     st.session_state.playlist_actions = {}

# # # # -------- CALLBACKS (unchanged) ---------
# # # def like_callback(song_key):
# # #     st.session_state.liked_songs.add(song_key)
# # #     st.session_state.disliked_songs.discard(song_key)

# # # def dislike_callback(song_key):
# # #     st.session_state.disliked_songs.add(song_key)
# # #     st.session_state.liked_songs.discard(song_key)

# # # def queue_callback(song_key):
# # #     st.session_state.queued_songs.add(song_key)

# # # def playlist_callback(song_key, rec):
# # #     st.session_state.playlist_actions[song_key] = {
# # #         'name': rec['name'],
# # #         'artist': rec['artist'],
# # #         'url': rec.get('spotify_preview_url', '')
# # #     }

# # # # DATA LOADING (unchanged)
# # # @st.cache_data
# # # def load_app_data():
# # #     try:
# # #         return {
# # #             'songs_data': pd.read_csv("data/cleaned_data.csv"),
# # #             'transformed_data': load_npz("data/transformed_data.npz"),
# # #             'track_ids': load("data/track_ids.npy", allow_pickle=True),
# # #             'filtered_data': pd.read_csv("data/collab_filtered_data.csv"),
# # #             'interaction_matrix': load_npz("data/interaction_matrix.npz"),
# # #             'transformed_hybrid_data': load_npz("data/transformed_hybrid_data.npz"),
# # #             'transformer': joblib_load('transformer.joblib')
# # #         }
# # #     except Exception as e:
# # #         st.error(f"Error loading data: {e}")
# # #         return None

# # # data = load_app_data()
# # # if data is None:
# # #     st.stop()

# # # songs_data = data['songs_data']
# # # transformed_data = data['transformed_data']
# # # track_ids = data['track_ids']
# # # filtered_data = data['filtered_data']
# # # interaction_matrix = data['interaction_matrix']
# # # transformed_hybrid_data = data['transformed_hybrid_data']
# # # transformer = data['transformer']

# # # # ---- MOODS (unchanged) ----
# # # MOOD_PRESETS = {
# # #     "😊 Happy": {"danceability": 0.8, "energy": 0.75, "valence": 0.9, "tempo": 130.0, "acousticness": 0.2, "instrumentalness": 0.1, "liveness": 0.3, "loudness": -5.0, "key": 2},
# # #     "😢 Sad": {"danceability": 0.3, "energy": 0.25, "valence": 0.15, "tempo": 75.0, "acousticness": 0.7, "instrumentalness": 0.3, "liveness": 0.1, "loudness": -15.0, "key": 9},
# # #     "🔥 Energetic": {"danceability": 0.9, "energy": 0.95, "valence": 0.8, "tempo": 150.0, "acousticness": 0.1, "instrumentalness": 0.05, "liveness": 0.4, "loudness": -3.0, "key": 7},
# # #     "😴 Relaxed": {"danceability": 0.4, "energy": 0.3, "valence": 0.6, "tempo": 85.0, "acousticness": 0.6, "instrumentalness": 0.4, "liveness": 0.15, "loudness": -12.0, "key": 5},
# # #     "💪 Workout": {"danceability": 0.85, "energy": 0.9, "valence": 0.75, "tempo": 140.0, "acousticness": 0.15, "instrumentalness": 0.1, "liveness": 0.3, "loudness": -4.0, "key": 1},
# # #     "🎉 Party": {"danceability": 0.95, "energy": 0.85, "valence": 0.85, "tempo": 125.0, "acousticness": 0.1, "instrumentalness": 0.05, "liveness": 0.5, "loudness": -3.5, "key": 4},
# # #     "🧘 Meditation": {"danceability": 0.2, "energy": 0.15, "valence": 0.65, "tempo": 60.0, "acousticness": 0.8, "instrumentalness": 0.7, "liveness": 0.05, "loudness": -20.0, "key": 3}
# # # }

# # # def get_mood_values(selected_mood):
# # #     if selected_mood == "Custom" or selected_mood not in MOOD_PRESETS:
# # #         return {"danceability": 0.5, "energy": 0.5, "valence": 0.5, "tempo": 120.0,
# # #                 "acousticness": 0.5, "instrumentalness": 0.5, "liveness": 0.2, "loudness": -10.0, "key": 0}
# # #     return MOOD_PRESETS[selected_mood]

# # # def recommend_by_available_features(user_preferences, songs_df, num_recs, exclude_songs=None):
# # #     try:
# # #         available_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'tempo', 'loudness', 'key']
# # #         existing_features = [f for f in available_features if f in songs_df.columns]
# # #         if not existing_features:
# # #             return pd.DataFrame()

# # #         filtered_songs = songs_df.copy()
# # #         if exclude_songs:
# # #             exclude_names = [song.split('_')[0] for song in exclude_songs]
# # #             filtered_songs = filtered_songs[~filtered_songs['name'].isin(exclude_names)]

# # #         user_vector = np.array([user_preferences.get(feature, 0.5) for feature in existing_features])
# # #         song_vectors = filtered_songs[existing_features].values
# # #         song_vectors = np.nan_to_num(song_vectors, nan=0.0)
# # #         user_vector_2d = user_vector.reshape(1, -1)
# # #         similarity_scores = cosine_similarity(user_vector_2d, song_vectors)[0]
# # #         top_indices = np.argsort(similarity_scores)[-num_recs:][::-1]
# # #         recommendations = filtered_songs.iloc[top_indices][['name', 'artist', 'spotify_preview_url']].copy()
# # #         recommendations['similarity_score'] = similarity_scores[top_indices]
# # #         return recommendations.reset_index(drop=True)
# # #     except Exception as e:
# # #         st.error(f"Error in recommendation: {e}")
# # #         return pd.DataFrame()

# # # # ---- IMPROVED UI Display Function ----
# # # def display_static_recommendations(recommendations, show_scores=False):
# # #     if recommendations.empty:
# # #         st.warning("🎵 No recommendations found. Try adjusting your preferences!")
# # #         return

# # #     st.session_state.current_recommendations = recommendations
    
# # #     for ind, rec in recommendations.iterrows():
# # #         song_name = rec['name'].title()
# # #         artist_name = rec['artist'].title()
# # #         song_key = f"{rec['name']}_{rec['artist']}"
        
# # #         # Song card container
# # #         st.markdown('<div class="song-card">', unsafe_allow_html=True)
        
# # #         # Playing indicators
# # #         if ind == 0:
# # #             st.markdown('<div class="now-playing">🎵 Currently Playing</div>', unsafe_allow_html=True)
# # #         elif ind == 1:
# # #             st.markdown('<div class="next-up">🎶 Next Up</div>', unsafe_allow_html=True)
        
# # #         # Song information
# # #         st.markdown(f'<div class="song-title">{song_name}</div>', unsafe_allow_html=True)
# # #         st.markdown(f'<div class="song-artist">by {artist_name}</div>', unsafe_allow_html=True)
        
# # #         if show_scores and 'similarity_score' in rec:
# # #             st.markdown(f'<div class="song-match">✨ Match: {rec["similarity_score"]:.1%}</div>', unsafe_allow_html=True)
        
# # #         # Status badges
# # #         status_html = ""
# # #         if song_key in st.session_state.liked_songs:
# # #             status_html += '<span class="status-badge badge-liked">💚 Liked</span>'
# # #         if song_key in st.session_state.disliked_songs:
# # #             status_html += '<span class="status-badge badge-disliked">💔 Disliked</span>'
# # #         if song_key in st.session_state.queued_songs:
# # #             status_html += '<span class="status-badge badge-queued">✅ Queued</span>'
        
# # #         if status_html:
# # #             st.markdown(status_html, unsafe_allow_html=True)
        
# # #         # Action buttons
# # #         col1, col2, col3, col4 = st.columns(4)
        
# # #         with col1:
# # #             if st.button("❤️ Like", key=f"like_{ind}_{song_key}"):
# # #                 like_callback(song_key)
# # #                 st.rerun()
        
# # #         with col2:
# # #             if st.button("👎 Pass", key=f"dislike_{ind}_{song_key}"):
# # #                 dislike_callback(song_key)
# # #                 st.rerun()
        
# # #         with col3:
# # #             if st.button("➕ Queue", key=f"queue_{ind}_{song_key}"):
# # #                 queue_callback(song_key)
# # #                 st.rerun()
        
# # #         with col4:
# # #             if st.button("📋 Playlist", key=f"playlist_{ind}_{song_key}"):
# # #                 playlist_callback(song_key, rec)
# # #                 st.rerun()
        
# # #         # Audio preview
# # #         st.markdown('<div class="audio-section">', unsafe_allow_html=True)
# # #         try:
# # #             if pd.notna(rec['spotify_preview_url']) and rec['spotify_preview_url'].strip():
# # #                 st.markdown("🔊 **Audio Preview Available**")
# # #                 st.audio(rec['spotify_preview_url'])
# # #             else:
# # #                 st.markdown("🔇 *Audio preview not available*")
# # #         except Exception:
# # #             st.markdown("🔇 *Audio preview not available*")
# # #         st.markdown('</div>', unsafe_allow_html=True)
        
# # #         st.markdown('</div>', unsafe_allow_html=True)
# # #         st.markdown("---")

# # # # --- Sidebar Navigation ---
# # # st.sidebar.markdown('<h1 style="color: white;">🎵 Navigation</h1>', unsafe_allow_html=True)
# # # page = st.sidebar.selectbox("Choose Page", [
# # #     "🏠 Home", "🔍 Discover", "📊 Analytics", "📝 Playlists", "⚙️ Settings"])

# # # # --- Main Home Page Logic ---
# # # if page == "🏠 Home":
# # #     # Main header
# # #     st.markdown('<h1 class="main-title">🎵 Spotify Song Recommender</h1>', unsafe_allow_html=True)
# # #     st.markdown('<div class="subtitle">🎧 Discover your next favorite song with AI-powered recommendations</div>', unsafe_allow_html=True)
    
# # #     # Algorithm selection
# # #     recommender_type = st.selectbox(
# # #         '🤖 Choose Your Algorithm:',
# # #         ['Feature Sliders', 'Content-Based Filtering', 'Collaborative Filtering', 'Hybrid Recommender System'],
# # #         index=0
# # #     )
    
# # #     enable_song_inputs = recommender_type != "Feature Sliders"
    
# # #     # Input fields
# # #     col1, col2 = st.columns(2)
# # #     with col1:
# # #         song_name = st.text_input('🎵 Enter a song name:', disabled=not enable_song_inputs)
# # #     with col2:
# # #         artist_name = st.text_input('🎤 Enter the artist name:', disabled=not enable_song_inputs)
    
# # #     col1, col2 = st.columns(2)
# # #     with col1:
# # #         k = st.selectbox('📊 Number of recommendations:', [5, 10, 15, 20], index=1)
# # #     with col2:
# # #         exclude_heard = st.checkbox("🚫 Exclude liked/disliked songs", value=True)
    
# # #     # Sidebar controls
# # #     st.sidebar.header("🎵 Audio Features")
# # #     st.sidebar.subheader("😊 Quick Mood Presets")
    
# # #     mood_options = ["Custom"] + list(MOOD_PRESETS.keys())
# # #     new_mood = st.sidebar.selectbox("Choose a mood:", mood_options, key="mood_selector")
    
# # #     if new_mood != st.session_state.selected_mood:
# # #         st.session_state.selected_mood = new_mood
# # #         if new_mood != "Custom":
# # #             st.rerun()
    
# # #     mood_values = get_mood_values(st.session_state.selected_mood)
    
# # #     if st.session_state.selected_mood != "Custom":
# # #         st.sidebar.markdown(f'<div class="mood-indicator">🎭 {st.session_state.selected_mood} Active</div>', unsafe_allow_html=True)
    
# # #     st.sidebar.subheader("🎛️ Core Features")
# # #     danceability = st.sidebar.slider('💃 Danceability', 0.0, 1.0, float(mood_values["danceability"]))
# # #     energy = st.sidebar.slider('⚡ Energy', 0.0, 1.0, float(mood_values["energy"]))
# # #     valence = st.sidebar.slider('😊 Valence', 0.0, 1.0, float(mood_values["valence"]))
# # #     tempo = st.sidebar.slider('🥁 Tempo', 60.0, 200.0, float(mood_values["tempo"]))
    
# # #     with st.sidebar.expander("🎛️ Advanced Features"):
# # #         acousticness = st.slider('🎸 Acousticness', 0.0, 1.0, float(mood_values["acousticness"]))
# # #         instrumentalness = st.slider('🎹 Instrumentalness', 0.0, 1.0, float(mood_values["instrumentalness"]))
# # #         liveness = st.slider('🎤 Liveness', 0.0, 1.0, float(mood_values["liveness"]))
# # #         loudness = st.slider('🔊 Loudness', -60.0, 0.0, float(mood_values["loudness"]))
# # #         key = st.selectbox('🎼 Key', list(range(12)), index=int(mood_values["key"]))
    
# # #     # Current preferences display
# # #     st.write("### 🎯 Current Preferences")
# # #     pref_col1, pref_col2, pref_col3, pref_col4 = st.columns(4)
    
# # #     with pref_col1:
# # #         st.markdown(f'''
# # #         <div class="metric-container">
# # #             <div class="metric-value">{danceability:.2f}</div>
# # #             <div class="metric-label">💃 Danceability</div>
# # #         </div>
# # #         ''', unsafe_allow_html=True)
    
# # #     with pref_col2:
# # #         st.markdown(f'''
# # #         <div class="metric-container">
# # #             <div class="metric-value">{energy:.2f}</div>
# # #             <div class="metric-label">⚡ Energy</div>
# # #         </div>
# # #         ''', unsafe_allow_html=True)
    
# # #     with pref_col3:
# # #         st.markdown(f'''
# # #         <div class="metric-container">
# # #             <div class="metric-value">{valence:.2f}</div>
# # #             <div class="metric-label">😊 Valence</div>
# # #         </div>
# # #         ''', unsafe_allow_html=True)
    
# # #     with pref_col4:
# # #         st.markdown(f'''
# # #         <div class="metric-container">
# # #             <div class="metric-value">{tempo:.0f}</div>
# # #             <div class="metric-label">🥁 BPM</div>
# # #         </div>
# # #         ''', unsafe_allow_html=True)
    
# # #     st.write("")  # Add some space
    
# # #     # Get recommendations button
# # #     if st.button('🎵 Get Recommendations', type="primary", use_container_width=True):
# # #         exclude_list = list(st.session_state.liked_songs) + list(st.session_state.disliked_songs) if exclude_heard else []
# # #         user_preferences = {
# # #             'danceability': danceability, 'energy': energy, 'valence': valence,
# # #             'acousticness': acousticness, 'instrumentalness': instrumentalness,
# # #             'liveness': liveness, 'tempo': tempo, 'loudness': loudness, 'key': key
# # #         }
        
# # #         if recommender_type == 'Feature Sliders':
# # #             recommendations = recommend_by_available_features(user_preferences, songs_data, k, exclude_list)
# # #             if not recommendations.empty:
# # #                 st.success(f'🎵 Found {len(recommendations)} songs for you!')
# # #                 display_static_recommendations(recommendations, show_scores=True)
# # #             else:
# # #                 st.warning("No songs found matching your criteria.")
# # #         else:
# # #             if not song_name or not artist_name:
# # #                 st.warning("Please enter BOTH song name and artist name for this algorithm.")
# # #             else:
# # #                 song_name_low = song_name.lower().strip()
# # #                 artist_name_low = artist_name.lower().strip()
                
# # #                 if recommender_type == 'Content-Based Filtering':
# # #                     if ((songs_data["name"] == song_name_low) & (songs_data['artist'] == artist_name_low)).any():
# # #                         recommendations = content_recommendation(song_name_low, artist_name_low, songs_data, transformed_data, k)
# # #                         st.success(f'🎵 Found songs similar to **{song_name}** by **{artist_name}**')
# # #                         display_static_recommendations(recommendations)
# # #                     else:
# # #                         st.warning("Song not found in database")
                
# # #                 elif recommender_type == 'Collaborative Filtering':
# # #                     if ((filtered_data["name"] == song_name_low) & (filtered_data['artist'] == artist_name_low)).any():
# # #                         recommendations = collaborative_recommendation(song_name_low, artist_name_low, track_ids, filtered_data, interaction_matrix, k)
# # #                         st.success(f'🎵 Collaborative recommendations for **{song_name}** by **{artist_name}**')
# # #                         display_static_recommendations(recommendations)
# # #                     else:
# # #                         st.warning("Song not found in collaborative dataset")
                
# # #                 elif recommender_type == 'Hybrid Recommender System':
# # #                     if ((filtered_data["name"] == song_name_low) & (filtered_data['artist'] == artist_name_low)).any():
# # #                         diversity = st.slider("🔀 Diversity", 1, 9, 5)
# # #                         content_based_weight = 1 - (diversity / 10)
# # #                         recommender = HybridRecommenderSystem(
# # #                             number_of_recommendations=k, weight_content_based=content_based_weight
# # #                         )
# # #                         recommendations = recommender.give_recommendations(
# # #                             song_name=song_name_low,
# # #                             artist_name=artist_name_low,
# # #                             songs_data=filtered_data,
# # #                             track_ids=track_ids,
# # #                             transformed_matrix=transformed_hybrid_data,
# # #                             interaction_matrix=interaction_matrix
# # #                         )
# # #                         st.success(f'🎵 Hybrid recommendations for **{song_name}** by **{artist_name}**')
# # #                         display_static_recommendations(recommendations)
# # #                     else:
# # #                         st.warning("Song not found in hybrid dataset")
    
# # #     elif not st.session_state.current_recommendations.empty:
# # #         st.write("### 🎵 Your Last Recommendations")
# # #         display_static_recommendations(st.session_state.current_recommendations, show_scores=True)

# # # # Other pages
# # # elif page == "🔍 Discover":
# # #     st.markdown('<h1 class="main-title">🔍 Advanced Discovery</h1>', unsafe_allow_html=True)
# # #     st.info("🚧 Advanced discovery features coming soon!")

# # # elif page == "📊 Analytics":
# # #     st.markdown('<h1 class="main-title">📊 Your Music Analytics</h1>', unsafe_allow_html=True)
# # #     st.info("📈 Analytics dashboard coming soon!")

# # # elif page == "📝 Playlists":
# # #     st.markdown('<h1 class="main-title">📝 My Playlists</h1>', unsafe_allow_html=True)
# # #     st.info("🎵 Playlist management features coming soon!")

# # # elif page == "⚙️ Settings":
# # #     st.markdown('<h1 class="main-title">⚙️ Settings</h1>', unsafe_allow_html=True)
# # #     st.info("⚙️ Settings panel coming soon!")

# # # # Footer
# # # st.markdown("---")
# # # st.markdown('<div style="text-align: center; color: #1DB954; font-weight: bold; padding: 1rem;">🎵 Spotify Song Recommender | Enhanced UI | Powered by AI</div>', unsafe_allow_html=True)













# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # from numpy import load
# # from scipy.sparse import load_npz
# # from joblib import load as joblib_load
# # from content_based_filtering import content_recommendation
# # from collaborative_filtering import collaborative_recommendation
# # from hybrid_recommendations import HybridRecommenderSystem
# # from sklearn.metrics.pairwise import cosine_similarity

# # # --------- CSS for modern design and spacing ---------
# # st.markdown("""
# # <style>
# # .main-header {
# #     font-size: 3rem; font-weight: bold; text-align: center;
# #     background: linear-gradient(90deg, #1DB954, #1ed760);
# #     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
# #     margin-bottom: 1rem;
# # }
# # .metric-card {
# #     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
# #     padding: 1.2rem; border-radius: 15px;
# #     color: white; text-align: center;
# #     box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
# #     margin-bottom: 0.8rem;
# # }
# # .song-card {
# #     background: linear-gradient(135deg, #fff 60%, #e2ffe9 100%);
# #     padding: 1.25rem 1.2rem 1.2rem 1.2rem; border-radius: 16px;
# #     margin: 1.1rem 0;
# #     border-left: 5px solid #1DB954;
# #     box-shadow: 0 5px 18px rgba(29, 185, 84, 0.10);
# # }
# # .song-title { font-size: 1.15rem; font-weight: 700; color: #1DB954; margin-bottom: .18em;}
# # .song-artist { font-size: 1rem;  color: #444;  margin-bottom: .4em;}
# # .song-match { font-size: 0.93rem; color: #22bb77; font-weight: 600; }
# # .status-badges { margin: 0.1rem 0 0.5rem 0;}
# # .status-badge {
# #     padding: 0.25rem 0.85rem; border-radius: 20px;
# #     font-size: 0.8rem; font-weight: 600; display:inline-block;
# #     margin-right: 0.5rem; margin-bottom:0.22em;
# # }
# # .badge-liked { background: #fff0f1; color: #ff6b6b;}
# # .badge-disliked { background: #e0f5ff; color: #3498db;}
# # .badge-queued { background: #f3f0ff; color: #6c5ce7; }
# # .now-playing {
# #     background: linear-gradient(90deg, #1DB954, #1ed760);
# #     color: white; padding: 0.45rem 1.2rem; border-radius: 18px;
# #     font-size: 0.99rem; font-weight: 500;
# #     margin-bottom: 0.65rem; display:inline-block;
# # }
# # .next-up {
# #     background: #eafaf2;
# #     color: #1DB954; padding: 0.36rem 1.05rem; border-radius: 17px;
# #     font-size: 0.92rem; font-weight: 600;
# #     margin-bottom: 0.52rem; display:inline-block;
# # }
# # .audio-preview { margin-top:0.7rem; }
# # .audio-available { color: #1DB954; font-weight: 500;}
# # .audio-unavailable { color: #9ca3af; font-style: italic;}
# # /* BUTTON: style ALL st.button globally for spacing and appearance */
# # .stButton > button {
# #     margin-right: 13px; margin-left: 2px;
# #     padding: 0.5rem 1.2rem !important;
# #     border-radius: 16px !important;
# #     background: linear-gradient(90deg, #1DB954, #1ed760) !important;
# #     color: white; font-weight: 600; font-size:1rem;
# #     border: none; box-shadow: 0 1.5px 7px rgba(29,185,84,.09);
# #     transition: all 0.18s;
# # }
# # .stButton > button:hover {
# #     background: linear-gradient(90deg, #38e44e, #4cffde) !important;
# #     color: #212121;
# # }
# # </style>
# # """, unsafe_allow_html=True)

# # # ------ SESSION STATE ------
# # if 'actions' not in st.session_state:
# #     st.session_state.actions = {}
# # if 'playlists' not in st.session_state:
# #     st.session_state.playlists = {}
# # if 'recommendation_history' not in st.session_state:
# #     st.session_state.recommendation_history = []
# # if 'user_preferences' not in st.session_state:
# #     st.session_state.user_preferences = {}
# # if 'liked_songs' not in st.session_state:
# #     st.session_state.liked_songs = set()
# # if 'disliked_songs' not in st.session_state:
# #     st.session_state.disliked_songs = set()
# # if 'queued_songs' not in st.session_state:
# #     st.session_state.queued_songs = set()
# # if 'selected_mood' not in st.session_state:
# #     st.session_state.selected_mood = "Custom"
# # if 'current_recommendations' not in st.session_state:
# #     st.session_state.current_recommendations = pd.DataFrame()
# # if 'playlist_actions' not in st.session_state:
# #     st.session_state.playlist_actions = {}

# # # -------- CALLBACKS FOR ACTION BUTTONS ---------
# # def like_callback(song_key):
# #     st.session_state.liked_songs.add(song_key)
# #     st.session_state.disliked_songs.discard(song_key)
# # def dislike_callback(song_key):
# #     st.session_state.disliked_songs.add(song_key)
# #     st.session_state.liked_songs.discard(song_key)
# # def queue_callback(song_key):
# #     st.session_state.queued_songs.add(song_key)
# # def playlist_callback(song_key, rec):
# #     st.session_state.playlist_actions[song_key] = {
# #         'name': rec['name'],
# #         'artist': rec['artist'],
# #         'url': rec.get('spotify_preview_url', '')
# #     }
# # def quick_add_callback(song_key, selected, rec):
# #     song_info = st.session_state.playlist_actions[song_key]
# #     if selected != "Select playlist...":
# #         exists = any(
# #             s['name'] == song_info['name'] and s['artist'] == song_info['artist']
# #             for s in st.session_state.playlists[selected]
# #         )
# #         if not exists:
# #             st.session_state.playlists[selected].append(song_info)
# #         del st.session_state.playlist_actions[song_key]
# # def quick_cancel_callback(song_key):
# #     if song_key in st.session_state.playlist_actions:
# #         del st.session_state.playlist_actions[song_key]

# # # DATA LOADING
# # @st.cache_data
# # def load_app_data():
# #     try:
# #         return {
# #             'songs_data': pd.read_csv("data/cleaned_data.csv"),
# #             'transformed_data': load_npz("data/transformed_data.npz"),
# #             'track_ids': load("data/track_ids.npy", allow_pickle=True),
# #             'filtered_data': pd.read_csv("data/collab_filtered_data.csv"),
# #             'interaction_matrix': load_npz("data/interaction_matrix.npz"),
# #             'transformed_hybrid_data': load_npz("data/transformed_hybrid_data.npz"),
# #             'transformer': joblib_load('transformer.joblib')
# #         }
# #     except Exception as e:
# #         st.error(f"Error loading data: {e}")
# #         return None
# # data = load_app_data()
# # if data is None:
# #     st.stop()
# # songs_data = data['songs_data']
# # transformed_data = data['transformed_data']
# # track_ids = data['track_ids']
# # filtered_data = data['filtered_data']
# # interaction_matrix = data['interaction_matrix']
# # transformed_hybrid_data = data['transformed_hybrid_data']
# # transformer = data['transformer']

# # # ---- MOODS ----
# # MOOD_PRESETS = {
# #     "😊 Happy": {"danceability": 0.8, "energy": 0.75, "valence": 0.9, "tempo": 130.0, "acousticness": 0.2, "instrumentalness": 0.1, "liveness": 0.3, "loudness": -5.0, "key": 2},
# #     "😢 Sad": {"danceability": 0.3, "energy": 0.25, "valence": 0.15, "tempo": 75.0, "acousticness": 0.7, "instrumentalness": 0.3, "liveness": 0.1, "loudness": -15.0, "key": 9},
# #     "🔥 Energetic": {"danceability": 0.9, "energy": 0.95, "valence": 0.8, "tempo": 150.0, "acousticness": 0.1, "instrumentalness": 0.05, "liveness": 0.4, "loudness": -3.0, "key": 7},
# #     "😴 Relaxed": {"danceability": 0.4, "energy": 0.3, "valence": 0.6, "tempo": 85.0, "acousticness": 0.6, "instrumentalness": 0.4, "liveness": 0.15, "loudness": -12.0, "key": 5},
# #     "💪 Workout": {"danceability": 0.85, "energy": 0.9, "valence": 0.75, "tempo": 140.0, "acousticness": 0.15, "instrumentalness": 0.1, "liveness": 0.3, "loudness": -4.0, "key": 1},
# #     "🎉 Party": {"danceability": 0.95, "energy": 0.85, "valence": 0.85, "tempo": 125.0, "acousticness": 0.1, "instrumentalness": 0.05, "liveness": 0.5, "loudness": -3.5, "key": 4},
# #     "🧘 Meditation": {"danceability": 0.2, "energy": 0.15, "valence": 0.65, "tempo": 60.0, "acousticness": 0.8, "instrumentalness": 0.7, "liveness": 0.05, "loudness": -20.0, "key": 3}
# # }
# # def get_mood_values(selected_mood):
# #     if selected_mood == "Custom" or selected_mood not in MOOD_PRESETS:
# #         return {"danceability": 0.5, "energy": 0.5, "valence": 0.5, "tempo": 120.0,
# #                 "acousticness": 0.5, "instrumentalness": 0.5, "liveness": 0.2, "loudness": -10.0, "key": 0}
# #     return MOOD_PRESETS[selected_mood]
# # def recommend_by_available_features(user_preferences, songs_df, num_recs, exclude_songs=None):
# #     try:
# #         available_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'tempo', 'loudness', 'key']
# #         existing_features = [f for f in available_features if f in songs_df.columns]
# #         if not existing_features:
# #             return pd.DataFrame()
# #         filtered_songs = songs_df.copy()
# #         if exclude_songs:
# #             exclude_names = [song.split('_')[0] for song in exclude_songs]
# #             filtered_songs = filtered_songs[~filtered_songs['name'].isin(exclude_names)]
# #         user_vector = np.array([user_preferences.get(feature, 0.5) for feature in existing_features])
# #         song_vectors = filtered_songs[existing_features].values
# #         song_vectors = np.nan_to_num(song_vectors, nan=0.0)
# #         user_vector_2d = user_vector.reshape(1, -1)
# #         similarity_scores = cosine_similarity(user_vector_2d, song_vectors)[0]
# #         top_indices = np.argsort(similarity_scores)[-num_recs:][::-1]
# #         recommendations = filtered_songs.iloc[top_indices][['name', 'artist', 'spotify_preview_url']].copy()
# #         recommendations['similarity_score'] = similarity_scores[top_indices]
# #         return recommendations.reset_index(drop=True)
# #     except Exception as e:
# #         st.error(f"Error in recommendation: {e}")
# #         return pd.DataFrame()

# # # ---- Modern display of recommendations ----
# # def display_static_recommendations(recommendations, show_scores=False):
# #     if recommendations.empty:
# #         st.warning("🎵 No recommendations found. Try adjusting your preferences!")
# #         return
# #     st.session_state.current_recommendations = recommendations
# #     for ind, rec in recommendations.iterrows():
# #         song_name = rec['name'].title()
# #         artist_name = rec['artist'].title()
# #         song_key = f"{rec['name']}_{rec['artist']}"
# #         st.markdown('<div class="song-card">', unsafe_allow_html=True)
        
# #         # Playing status
# #         if ind == 0:
# #             st.markdown('<div class="now-playing">🎵 Currently Playing</div>', unsafe_allow_html=True)
# #         elif ind == 1:
# #             st.markdown('<div class="next-up">🎶 Next Up</div>', unsafe_allow_html=True)
        
# #         st.markdown(f'<div class="song-title">{song_name}</div>', unsafe_allow_html=True)
# #         st.markdown(f'<div class="song-artist">by {artist_name}</div>', unsafe_allow_html=True)
# #         if show_scores and 'similarity_score' in rec:
# #             st.markdown(f'<div class="song-match">✨ Match: {rec["similarity_score"]:.1%}</div>', unsafe_allow_html=True)
        
# #         # Status badges
# #         status_html = '<div class="status-badges">'
# #         if song_key in st.session_state.liked_songs:
# #             status_html += '<span class="status-badge badge-liked">💚 Liked</span>'
# #         if song_key in st.session_state.disliked_songs:
# #             status_html += '<span class="status-badge badge-disliked">💔 Disliked</span>'
# #         if song_key in st.session_state.queued_songs:
# #             status_html += '<span class="status-badge badge-queued">✅ Queued</span>'
# #         status_html += '</div>'
# #         st.markdown(status_html, unsafe_allow_html=True)

# #         cols = st.columns(4)
# #         if cols[0].button("❤️ Like", key=f"like_{ind}_{song_key}"):
# #             like_callback(song_key); st.rerun()
# #         if cols[1].button("👎 Pass", key=f"dislike_{ind}_{song_key}"):
# #             dislike_callback(song_key); st.rerun()
# #         if cols[2].button("➕ Queue", key=f"queue_{ind}_{song_key}"):
# #             queue_callback(song_key); st.rerun()
# #         if cols[3].button("📋 Playlist", key=f"playlist_{ind}_{song_key}"):
# #             playlist_callback(song_key, rec); st.rerun()

# #         st.markdown('<div class="audio-preview">', unsafe_allow_html=True)
# #         try:
# #             if pd.notna(rec['spotify_preview_url']) and rec['spotify_preview_url'].strip():
# #                 st.markdown('<div class="audio-available">🔊 Audio Preview Available</div>', unsafe_allow_html=True)
# #                 st.audio(rec['spotify_preview_url'])
# #             else:
# #                 st.markdown('<div class="audio-unavailable">🔇 Audio preview not available</div>', unsafe_allow_html=True)
# #         except Exception:
# #             st.markdown('<div class="audio-unavailable">🔇 Audio preview not available</div>', unsafe_allow_html=True)
# #         st.markdown('</div>', unsafe_allow_html=True)
# #         st.markdown('</div>', unsafe_allow_html=True)
# #         st.markdown('<br>', unsafe_allow_html=True)

# # # --- Sidebar Navigation ---
# # st.sidebar.markdown('<h1 style="color: #1DB954;">🎵 Navigation</h1>', unsafe_allow_html=True)
# # page = st.sidebar.selectbox("Choose Page", [
# #     "🏠 Home", "🔍 Discover", "📊 Analytics", "📝 Playlists", "⚙️ Settings"
# # ])

# # # --- Main Home Page Logic ---
# # if page == "🏠 Home":
# #     st.markdown('<h1 class="main-header">🎵 Spotify Song Recommender</h1>', unsafe_allow_html=True)
# #     st.markdown('<div style="text-align: center; color: #555; font-size:1.25rem; margin-bottom:2rem;">🎧 Discover your next favorite song with AI-powered recommendations</div>', unsafe_allow_html=True)
    
# #     recommender_type = st.selectbox(
# #         '🤖 Choose Your Algorithm:',
# #         ['Feature Sliders', 'Content-Based Filtering', 'Collaborative Filtering', 'Hybrid Recommender System'],
# #         index=0)
# #     enable_song_inputs = recommender_type != "Feature Sliders"
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         song_name = st.text_input('🎵 Enter a song name:', disabled=not enable_song_inputs)
# #     with col2:
# #         artist_name = st.text_input('🎤 Enter the artist name:', disabled=not enable_song_inputs)
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         k = st.selectbox('📊 Number of recommendations:', [5, 10, 15, 20], index=1)
# #     with col2:
# #         exclude_heard = st.checkbox("🚫 Exclude liked/disliked songs", value=True)

# #     # Enhanced sidebar controls
# #     st.sidebar.markdown('<h2 style="color: #1DB954; margin-top: 2rem;">🎵 Audio Features</h2>', unsafe_allow_html=True)
# #     st.sidebar.markdown('<h3 style="color: #1DB954; font-size: 1.1rem;">😊 Quick Mood Presets</h3>', unsafe_allow_html=True)

# #     mood_options = ["Custom"] + list(MOOD_PRESETS.keys())
# #     new_mood = st.sidebar.selectbox("Choose a mood:", mood_options, key="mood_selector")
# #     if new_mood != st.session_state.selected_mood:
# #         st.session_state.selected_mood = new_mood
# #         if new_mood != "Custom":
# #             st.rerun()
# #     mood_values = get_mood_values(st.session_state.selected_mood)
# #     if st.session_state.selected_mood != "Custom":
# #         st.sidebar.markdown(f'<div style="background: linear-gradient(90deg,#1DB954,#1ed760); color:white; font-weight:600; border-radius:16px; display:inline-block; padding:0.18rem 1.2rem;">🎭 {st.session_state.selected_mood} Active</div>', unsafe_allow_html=True)

# #     st.sidebar.markdown('<h3 style="color: #1DB954; font-size: 1.1rem; margin-top:1.3rem;">🎛️ Core Features</h3>', unsafe_allow_html=True)
# #     danceability = st.sidebar.slider('💃 Danceability', 0.0, 1.0, float(mood_values["danceability"]))
# #     energy = st.sidebar.slider('⚡ Energy', 0.0, 1.0, float(mood_values["energy"]))
# #     valence = st.sidebar.slider('😊 Valence', 0.0, 1.0, float(mood_values["valence"]))
# #     tempo = st.sidebar.slider('🥁 Tempo', 60.0, 200.0, float(mood_values["tempo"]))

# #     with st.sidebar.expander("🎛️ Advanced Features", expanded=False):
# #         acousticness = st.slider('🎸 Acousticness', 0.0, 1.0, float(mood_values["acousticness"]))
# #         instrumentalness = st.slider('🎹 Instrumentalness', 0.0, 1.0, float(mood_values["instrumentalness"]))
# #         liveness = st.slider('🎤 Liveness', 0.0, 1.0, float(mood_values["liveness"]))
# #         loudness = st.slider('🔊 Loudness', -60.0, 0.0, float(mood_values["loudness"]))
# #         key = st.selectbox('🎼 Key', list(range(12)), index=int(mood_values["key"]))

# #     # Preferences display
# #     st.markdown("### 🎯 Current Preferences")
# #     pref_col1, pref_col2, pref_col3, pref_col4 = st.columns(4)
# #     with pref_col1:
# #         st.markdown(f'''
# #         <div class="metric-card">
# #             <div style="font-size:2rem; font-weight:700;">{danceability:.2f}</div>
# #             <div style="font-size:0.9rem; opacity: 0.95; font-weight:600;">💃 Danceability</div>
# #         </div>
# #         ''', unsafe_allow_html=True)
# #     with pref_col2:
# #         st.markdown(f'''
# #         <div class="metric-card">
# #             <div style="font-size:2rem; font-weight:700;">{energy:.2f}</div>
# #             <div style="font-size:0.9rem; opacity: 0.95; font-weight:600;">⚡ Energy</div>
# #         </div>
# #         ''', unsafe_allow_html=True)
# #     with pref_col3:
# #         st.markdown(f'''
# #         <div class="metric-card">
# #             <div style="font-size:2rem; font-weight:700;">{valence:.2f}</div>
# #             <div style="font-size:0.9rem; opacity: 0.95; font-weight:600;">😊 Valence</div>
# #         </div>
# #         ''', unsafe_allow_html=True)
# #     with pref_col4:
# #         st.markdown(f'''
# #         <div class="metric-card">
# #             <div style="font-size:2rem; font-weight:700;">{tempo:.0f}</div>
# #             <div style="font-size:0.9rem; opacity: 0.95; font-weight:600;">🥁 BPM</div>
# #         </div>
# #         ''', unsafe_allow_html=True)
# #     st.markdown('<br>', unsafe_allow_html=True)

# #     if st.button('🎵 Get Recommendations', type="primary", use_container_width=True):
# #         with st.spinner('🎵 Finding your perfect songs...'):
# #             exclude_list = list(st.session_state.liked_songs) + list(st.session_state.disliked_songs) if exclude_heard else []
# #             user_preferences = {
# #                 'danceability': danceability, 'energy': energy, 'valence': valence,
# #                 'acousticness': acousticness, 'instrumentalness': instrumentalness,
# #                 'liveness': liveness, 'tempo': tempo, 'loudness': loudness, 'key': key
# #             }
# #             if recommender_type == 'Feature Sliders':
# #                 recommendations = recommend_by_available_features(user_preferences, songs_data, k, exclude_list)
# #                 if not recommendations.empty:
# #                     st.success(f'🎵 Found {len(recommendations)} amazing songs for you!')
# #                     display_static_recommendations(recommendations, show_scores=True)
# #                 else:
# #                     st.warning("🎵 No songs found matching your criteria. Try adjusting your preferences!")
# #             else:
# #                 if not song_name or not artist_name:
# #                     st.warning("⚠️ Please enter BOTH song name and artist name for this algorithm.")
# #                 else:
# #                     song_name_low = song_name.lower().strip()
# #                     artist_name_low = artist_name.lower().strip()
# #                     if recommender_type == 'Content-Based Filtering':
# #                         if ((songs_data["name"] == song_name_low) & (songs_data['artist'] == artist_name_low)).any():
# #                             recommendations = content_recommendation(song_name_low, artist_name_low, songs_data, transformed_data, k)
# #                             st.success(f'🎵 Found songs similar to **{song_name}** by **{artist_name}**!')
# #                             display_static_recommendations(recommendations)
# #                         else:
# #                             st.error("❌ Song not found in database. Please check the spelling and try again.")
# #                     elif recommender_type == 'Collaborative Filtering':
# #                         if ((filtered_data["name"] == song_name_low) & (filtered_data['artist'] == artist_name_low)).any():
# #                             recommendations = collaborative_recommendation(song_name_low, artist_name_low, track_ids, filtered_data, interaction_matrix, k)
# #                             st.success(f'🎵 Collaborative recommendations for **{song_name}** by **{artist_name}**!')
# #                             display_static_recommendations(recommendations)
# #                         else:
# #                             st.error("❌ Song not found in collaborative dataset. Try a different song.")
# #                     elif recommender_type == 'Hybrid Recommender System':
# #                         if ((filtered_data["name"] == song_name_low) & (filtered_data['artist'] == artist_name_low)).any():
# #                             diversity = st.slider("🔀 Diversity Level", 1, 9, 5, help="Higher values = more diverse recommendations")
# #                             content_based_weight = 1 - (diversity / 10)
# #                             recommender = HybridRecommenderSystem(
# #                                 number_of_recommendations=k, weight_content_based=content_based_weight
# #                             )
# #                             recommendations = recommender.give_recommendations(
# #                                 song_name=song_name_low,
# #                                 artist_name=artist_name_low,
# #                                 songs_data=filtered_data,
# #                                 track_ids=track_ids,
# #                                 transformed_matrix=transformed_hybrid_data,
# #                                 interaction_matrix=interaction_matrix
# #                             )
# #                             st.success(f'🎵 Hybrid recommendations for **{song_name}** by **{artist_name}**!')
# #                             display_static_recommendations(recommendations)
# #                         else:
# #                             st.error("❌ Song not found in hybrid dataset. Try a different song.")
# #     elif not st.session_state.current_recommendations.empty:
# #         st.markdown("### 🎵 Your Last Recommendations")
# #         display_static_recommendations(st.session_state.current_recommendations, show_scores=True)

# # elif page == "🔍 Discover":
# #     st.markdown('<h1 class="main-header">🔍 Advanced Discovery</h1>', unsafe_allow_html=True)
# #     st.info("🚧 Advanced discovery features coming soon!")

# # elif page == "📊 Analytics":
# #     st.markdown('<h1 class="main-header">📊 Your Music Analytics</h1>', unsafe_allow_html=True)
# #     st.info("📈 Analytics dashboard coming soon!")

# # elif page == "📝 Playlists":
# #     st.markdown('<h1 class="main-header">📝 My Playlists</h1>', unsafe_allow_html=True)
# #     st.info("🎵 Playlist management features coming soon!")

# # elif page == "⚙️ Settings":
# #     st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
# #     st.info("⚙️ Settings panel coming soon!")

# # st.markdown('''
# # <div style="text-align: center; padding: 1.5rem 0 0.5rem 0; color: #1DB954; font-weight: bold; font-size:1.18rem;">
# #     🎵 Spotify Song Recommender | Modern UI | Powered by AI
# #     <br>
# #     <span style="font-size:0.92rem;color:#222;">Discover • Enjoy • Share</span>
# # </div>
# # ''', unsafe_allow_html=True)




# import streamlit as st
# import pandas as pd
# import numpy as np
# from numpy import load
# from scipy.sparse import load_npz
# from joblib import load as joblib_load
# from content_based_filtering import content_recommendation
# from collaborative_filtering import collaborative_recommendation
# from hybrid_recommendations import HybridRecommenderSystem
# from sklearn.metrics.pairwise import cosine_similarity

# # --------- CSS for modern design and spacing ---------
# st.markdown("""
# <style>
# ... (same style as before) ...
# </style>
# """, unsafe_allow_html=True)

# # ------ SESSION STATE ------
# if 'actions' not in st.session_state:
#     st.session_state.actions = {}
# if 'playlists' not in st.session_state:
#     st.session_state.playlists = {}
# if 'recommendation_history' not in st.session_state:
#     st.session_state.recommendation_history = []
# if 'user_preferences' not in st.session_state:
#     st.session_state.user_preferences = {}
# if 'liked_songs' not in st.session_state:
#     st.session_state.liked_songs = set()
# if 'disliked_songs' not in st.session_state:
#     st.session_state.disliked_songs = set()
# if 'queued_songs' not in st.session_state:
#     st.session_state.queued_songs = set()
# if 'selected_mood' not in st.session_state:
#     st.session_state.selected_mood = "Custom"
# if 'current_recommendations' not in st.session_state:
#     st.session_state.current_recommendations = pd.DataFrame()
# if 'playlist_actions' not in st.session_state:
#     st.session_state.playlist_actions = {}

# # -------- CALLBACKS FOR ACTION BUTTONS ---------
# def like_callback(song_key):
#     st.session_state.liked_songs.add(song_key)
#     st.session_state.disliked_songs.discard(song_key)
# def dislike_callback(song_key):
#     st.session_state.disliked_songs.add(song_key)
#     st.session_state.liked_songs.discard(song_key)
# def queue_callback(song_key):
#     st.session_state.queued_songs.add(song_key)
# def playlist_callback(song_key, rec):
#     st.session_state.playlist_actions[song_key] = {
#         'name': rec['name'],
#         'artist': rec['artist'],
#         'url': rec.get('spotify_preview_url', '')
#     }

# # Unified action dispatcher to avoid unnecessary full reruns
# def handle_song_action(action, song_key, rec=None):
#     if action == "like":
#         like_callback(song_key)
#     elif action == "dislike":
#         dislike_callback(song_key)
#     elif action == "queue":
#         queue_callback(song_key)
#     elif action == "playlist" and rec is not None:
#         playlist_callback(song_key, rec)

# # DATA LOADING
# @st.cache_data
# def load_app_data():
#     try:
#         return {
#             'songs_data': pd.read_csv("data/cleaned_data.csv"),
#             'transformed_data': load_npz("data/transformed_data.npz"),
#             'track_ids': load("data/track_ids.npy", allow_pickle=True),
#             'filtered_data': pd.read_csv("data/collab_filtered_data.csv"),
#             'interaction_matrix': load_npz("data/interaction_matrix.npz"),
#             'transformed_hybrid_data': load_npz("data/transformed_hybrid_data.npz"),
#             'transformer': joblib_load('transformer.joblib')
#         }
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         return None
# data = load_app_data()
# if data is None:
#     st.stop()
# songs_data = data['songs_data']
# transformed_data = data['transformed_data']
# track_ids = data['track_ids']
# filtered_data = data['filtered_data']
# interaction_matrix = data['interaction_matrix']
# transformed_hybrid_data = data['transformed_hybrid_data']
# transformer = data['transformer']

# # ---- MOODS ----
# MOOD_PRESETS = {
#     "😊 Happy": {"danceability": 0.8, "energy": 0.75, "valence": 0.9, "tempo": 130.0, "acousticness": 0.2,"instrumentalness": 0.1,"liveness": 0.3,"loudness": -5.0,"key": 2},
#     "😢 Sad": {"danceability": 0.3,"energy": 0.25,"valence": 0.15,"tempo": 75.0,"acousticness": 0.7,"instrumentalness": 0.3,"liveness": 0.1,"loudness": -15.0,"key": 9},
#     "🔥 Energetic": {"danceability": 0.9,"energy": 0.95,"valence": 0.8,"tempo": 150.0,"acousticness": 0.1,"instrumentalness": 0.05,"liveness": 0.4,"loudness": -3.0,"key": 7},
#     "😴 Relaxed": {"danceability": 0.4,"energy": 0.3,"valence": 0.6,"tempo": 85.0,"acousticness": 0.6,"instrumentalness": 0.4,"liveness": 0.15,"loudness": -12.0,"key": 5},
#     "💪 Workout": {"danceability": 0.85,"energy": 0.9,"valence": 0.75,"tempo": 140.0,"acousticness": 0.15,"instrumentalness": 0.1,"liveness": 0.3,"loudness": -4.0,"key": 1},
#     "🎉 Party": {"danceability": 0.95,"energy": 0.85,"valence": 0.85,"tempo": 125.0,"acousticness": 0.1,"instrumentalness": 0.05,"liveness": 0.5,"loudness": -3.5,"key": 4},
#     "🧘 Meditation": {"danceability": 0.2,"energy": 0.15,"valence": 0.65,"tempo": 60.0,"acousticness": 0.8,"instrumentalness": 0.7,"liveness": 0.05,"loudness": -20.0,"key": 3}
# }
# def get_mood_values(selected_mood):
#     if selected_mood == "Custom" or selected_mood not in MOOD_PRESETS:
#         return {"danceability": 0.5,"energy": 0.5,"valence": 0.5,"tempo": 120.0,"acousticness": 0.5,"instrumentalness": 0.5,"liveness": 0.2,"loudness": -10.0,"key": 0}
#     return MOOD_PRESETS[selected_mood]
# def recommend_by_available_features(user_preferences, songs_df, num_recs, exclude_songs=None):
#     try:
#         available_features = ['danceability','energy','valence','acousticness','instrumentalness','liveness','tempo','loudness','key']
#         existing_features = [f for f in available_features if f in songs_df.columns]
#         if not existing_features: return pd.DataFrame()
#         filtered_songs = songs_df.copy()
#         if exclude_songs:
#             exclude_names = [song.split('_')[0] for song in exclude_songs]
#             filtered_songs = filtered_songs[~filtered_songs['name'].isin(exclude_names)]
#         user_vector = np.array([user_preferences.get(feature, 0.5) for feature in existing_features])
#         song_vectors = filtered_songs[existing_features].values
#         song_vectors = np.nan_to_num(song_vectors, nan=0.0)
#         user_vector_2d = user_vector.reshape(1, -1)
#         similarity_scores = cosine_similarity(user_vector_2d, song_vectors)[0]
#         top_indices = np.argsort(similarity_scores)[-num_recs:][::-1]
#         recommendations = filtered_songs.iloc[top_indices][['name','artist','spotify_preview_url']].copy()
#         recommendations['similarity_score'] = similarity_scores[top_indices]
#         return recommendations.reset_index(drop=True)
#     except Exception as e:
#         st.error(f"Error in recommendation: {e}")
#         return pd.DataFrame()

# # ---- Modern display of recommendations ----
# def display_static_recommendations(recommendations, show_scores=False):
#     if recommendations.empty:
#         st.warning("🎵 No recommendations found. Try adjusting your preferences!")
#         return
#     st.session_state.current_recommendations = recommendations
#     for ind, rec in recommendations.iterrows():
#         song_name = rec['name'].title()
#         artist_name = rec['artist'].title()
#         song_key = f"{rec['name']}_{rec['artist']}"
#         st.markdown('<div class="song-card">', unsafe_allow_html=True)
#         # Playing status
#         if ind == 0:
#             st.markdown('<div class="now-playing">🎵 Currently Playing</div>', unsafe_allow_html=True)
#         elif ind == 1:
#             st.markdown('<div class="next-up">🎶 Next Up</div>', unsafe_allow_html=True)
#         st.markdown(f'<div class="song-title">{song_name}</div>', unsafe_allow_html=True)
#         st.markdown(f'<div class="song-artist">by {artist_name}</div>', unsafe_allow_html=True)
#         if show_scores and 'similarity_score' in rec:
#             st.markdown(f'<div class="song-match">✨ Match: {rec["similarity_score"]:.1%}</div>', unsafe_allow_html=True)
#         # Status badges
#         status_html = '<div class="status-badges">'
#         if song_key in st.session_state.liked_songs:
#             status_html += '<span class="status-badge badge-liked">💚 Liked</span>'
#         if song_key in st.session_state.disliked_songs:
#             status_html += '<span class="status-badge badge-disliked">💔 Disliked</span>'
#         if song_key in st.session_state.queued_songs:
#             status_html += '<span class="status-badge badge-queued">✅ Queued</span>'
#         status_html += '</div>'
#         st.markdown(status_html, unsafe_allow_html=True)
#         # ACTION BUTTONS - set unique key for button group and use st.form for local rerun
#         with st.form(key=f'form_{ind}_{song_key}'):
#             cols = st.columns(4)
#             like = cols[0].form_submit_button("❤️ Like")
#             dislike = cols[1].form_submit_button("👎 Pass")
#             queue = cols[2].form_submit_button("➕ Queue")
#             playlist = cols[3].form_submit_button("📋 Playlist")
#             if like:
#                 handle_song_action("like", song_key)
#             if dislike:
#                 handle_song_action("dislike", song_key)
#             if queue:
#                 handle_song_action("queue", song_key)
#             if playlist:
#                 handle_song_action("playlist", song_key, rec)
#         # Audio preview
#         st.markdown('<div class="audio-preview">', unsafe_allow_html=True)
#         try:
#             if pd.notna(rec['spotify_preview_url']) and rec['spotify_preview_url'].strip():
#                 st.markdown('<div class="audio-available">🔊 Audio Preview Available</div>', unsafe_allow_html=True)
#                 st.audio(rec['spotify_preview_url'])
#             else:
#                 st.markdown('<div class="audio-unavailable">🔇 Audio preview not available</div>', unsafe_allow_html=True)
#         except Exception:
#             st.markdown('<div class="audio-unavailable">🔇 Audio preview not available</div>', unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)
#         st.markdown('<br>', unsafe_allow_html=True)

# # --- Sidebar Navigation ---
# st.sidebar.markdown('<h1 style="color: #1DB954;">🎵 Navigation</h1>', unsafe_allow_html=True)
# page = st.sidebar.selectbox("Choose Page", [
#     "🏠 Home", "🔍 Discover", "📊 Analytics", "📝 Playlists", "⚙️ Settings"
# ])

# # --- Main Home Page Logic ---
# if page == "🏠 Home":
#     st.markdown('<h1 class="main-header">🎵 Spotify Song Recommender</h1>', unsafe_allow_html=True)
#     st.markdown('<div style="text-align: center; color: #555; font-size:1.25rem; margin-bottom:2rem;">🎧 Discover your next favorite song with AI-powered recommendations</div>', unsafe_allow_html=True)
#     recommender_type = st.selectbox(
#         '🤖 Choose Your Algorithm:',
#         ['Feature Sliders', 'Content-Based Filtering', 'Collaborative Filtering', 'Hybrid Recommender System'],
#         index=0)
#     enable_song_inputs = recommender_type != "Feature Sliders"
#     col1, col2 = st.columns(2)
#     with col1:
#         song_name = st.text_input('🎵 Enter a song name:', disabled=not enable_song_inputs)
#     with col2:
#         artist_name = st.text_input('🎤 Enter the artist name:', disabled=not enable_song_inputs)
#     col1, col2 = st.columns(2)
#     with col1:
#         k = st.selectbox('📊 Number of recommendations:', [5, 10, 15, 20], index=1)
#     with col2:
#         exclude_heard = st.checkbox("🚫 Exclude liked/disliked songs", value=True)

#     # Enhanced sidebar controls
#     st.sidebar.markdown('<h2 style="color: #1DB954; margin-top: 2rem;">🎵 Audio Features</h2>', unsafe_allow_html=True)
#     st.sidebar.markdown('<h3 style="color: #1DB954; font-size: 1.1rem;">😊 Quick Mood Presets</h3>', unsafe_allow_html=True)
#     mood_options = ["Custom"] + list(MOOD_PRESETS.keys())
#     new_mood = st.sidebar.selectbox("Choose a mood:", mood_options, key="mood_selector")
#     if new_mood != st.session_state.selected_mood:
#         st.session_state.selected_mood = new_mood
#         if new_mood != "Custom":
#             st.experimental_rerun()
#     mood_values = get_mood_values(st.session_state.selected_mood)
#     if st.session_state.selected_mood != "Custom":
#         st.sidebar.markdown(f'<div style="background: linear-gradient(90deg,#1DB954,#1ed760); color:white; font-weight:600; border-radius:16px; display:inline-block; padding:0.18rem 1.2rem;">🎭 {st.session_state.selected_mood} Active</div>', unsafe_allow_html=True)
#     st.sidebar.markdown('<h3 style="color: #1DB954; font-size: 1.1rem; margin-top:1.3rem;">🎛️ Core Features</h3>', unsafe_allow_html=True)
#     danceability = st.sidebar.slider('💃 Danceability', 0.0, 1.0, float(mood_values["danceability"]))
#     energy = st.sidebar.slider('⚡ Energy', 0.0, 1.0, float(mood_values["energy"]))
#     valence = st.sidebar.slider('😊 Valence', 0.0, 1.0, float(mood_values["valence"]))
#     tempo = st.sidebar.slider('🥁 Tempo', 60.0, 200.0, float(mood_values["tempo"]))
#     with st.sidebar.expander("🎛️ Advanced Features", expanded=False):
#         acousticness = st.slider('🎸 Acousticness',0.0,1.0,float(mood_values["acousticness"]))
#         instrumentalness = st.slider('🎹 Instrumentalness',0.0,1.0,float(mood_values["instrumentalness"]))
#         liveness = st.slider('🎤 Liveness',0.0,1.0,float(mood_values["liveness"]))
#         loudness = st.slider('🔊 Loudness',-60.0,0.0,float(mood_values["loudness"]))
#         key = st.selectbox('🎼 Key',list(range(12)), index=int(mood_values["key"]))

#     # Preferences display
#     st.markdown("### 🎯 Current Preferences")
#     pref_col1, pref_col2, pref_col3, pref_col4 = st.columns(4)
#     with pref_col1:
#         st.markdown(f'''
#         <div class="metric-card">
#             <div style="font-size:2rem; font-weight:700;">{danceability:.2f}</div>
#             <div style="font-size:0.9rem; opacity: 0.95; font-weight:600;">💃 Danceability</div>
#         </div>
#         ''', unsafe_allow_html=True)
#     with pref_col2:
#         st.markdown(f'''
#         <div class="metric-card">
#             <div style="font-size:2rem; font-weight:700;">{energy:.2f}</div>
#             <div style="font-size:0.9rem; opacity: 0.95; font-weight:600;">⚡ Energy</div>
#         </div>
#         ''', unsafe_allow_html=True)
#     with pref_col3:
#         st.markdown(f'''
#         <div class="metric-card">
#             <div style="font-size:2rem; font-weight:700;">{valence:.2f}</div>
#             <div style="font-size:0.9rem; opacity: 0.95; font-weight:600;">😊 Valence</div>
#         </div>
#         ''', unsafe_allow_html=True)
#     with pref_col4:
#         st.markdown(f'''
#         <div class="metric-card">
#             <div style="font-size:2rem; font-weight:700;">{tempo:.0f}</div>
#             <div style="font-size:0.9rem; opacity: 0.95; font-weight:600;">🥁 BPM</div>
#         </div>
#         ''', unsafe_allow_html=True)
#     st.markdown('<br>', unsafe_allow_html=True)

#     if st.button('🎵 Get Recommendations', type="primary", use_container_width=True):
#         with st.spinner('🎵 Finding your perfect songs...'):
#             exclude_list = list(st.session_state.liked_songs) + list(st.session_state.disliked_songs) if exclude_heard else []
#             user_preferences = {
#                 'danceability': danceability, 'energy': energy, 'valence': valence,
#                 'acousticness': acousticness, 'instrumentalness': instrumentalness,
#                 'liveness': liveness, 'tempo': tempo, 'loudness': loudness, 'key': key
#             }
#             if recommender_type == 'Feature Sliders':
#                 recommendations = recommend_by_available_features(user_preferences, songs_data, k, exclude_list)
#                 if not recommendations.empty:
#                     st.success(f'🎵 Found {len(recommendations)} amazing songs for you!')
#                     display_static_recommendations(recommendations, show_scores=True)
#                 else:
#                     st.warning("🎵 No songs found matching your criteria. Try adjusting your preferences!")
#             else:
#                 if not song_name or not artist_name:
#                     st.warning("⚠️ Please enter BOTH song name and artist name for this algorithm.")
#                 else:
#                     song_name_low = song_name.lower().strip()
#                     artist_name_low = artist_name.lower().strip()
#                     if recommender_type == 'Content-Based Filtering':
#                         if ((songs_data["name"] == song_name_low) & (songs_data['artist'] == artist_name_low)).any():
#                             recommendations = content_recommendation(song_name_low, artist_name_low, songs_data, transformed_data, k)
#                             st.success(f'🎵 Found songs similar to **{song_name}** by **{artist_name}**!')
#                             display_static_recommendations(recommendations)
#                         else:
#                             st.error("❌ Song not found in database. Please check the spelling and try again.")
#                     elif recommender_type == 'Collaborative Filtering':
#                         if ((filtered_data["name"] == song_name_low) & (filtered_data['artist'] == artist_name_low)).any():
#                             recommendations = collaborative_recommendation(song_name_low, artist_name_low, track_ids, filtered_data, interaction_matrix, k)
#                             st.success(f'🎵 Collaborative recommendations for **{song_name}** by **{artist_name}**!')
#                             display_static_recommendations(recommendations)
#                         else:
#                             st.error("❌ Song not found in collaborative dataset. Try a different song.")
#                     elif recommender_type == 'Hybrid Recommender System':
#                         if ((filtered_data["name"] == song_name_low) & (filtered_data['artist'] == artist_name_low)).any():
#                             diversity = st.slider("🔀 Diversity Level", 1, 9, 5, help="Higher values = more diverse recommendations")
#                             content_based_weight = 1 - (diversity / 10)
#                             recommender = HybridRecommenderSystem(
#                                 number_of_recommendations=k, weight_content_based=content_based_weight
#                             )
#                             recommendations = recommender.give_recommendations(
#                                 song_name=song_name_low,
#                                 artist_name=artist_name_low,
#                                 songs_data=filtered_data,
#                                 track_ids=track_ids,
#                                 transformed_matrix=transformed_hybrid_data,
#                                 interaction_matrix=interaction_matrix
#                             )
#                             st.success(f'🎵 Hybrid recommendations for **{song_name}** by **{artist_name}**!')
#                             display_static_recommendations(recommendations)
#                         else:
#                             st.error("❌ Song not found in hybrid dataset. Try a different song.")
#     elif not st.session_state.current_recommendations.empty:
#         st.markdown("### 🎵 Your Last Recommendations")
#         display_static_recommendations(st.session_state.current_recommendations, show_scores=True)

# elif page == "🔍 Discover":
#     st.markdown('<h1 class="main-header">🔍 Advanced Discovery</h1>', unsafe_allow_html=True)
#     st.info("🚧 Advanced discovery features coming soon!")

# elif page == "📊 Analytics":
#     st.markdown('<h1 class="main-header">📊 Your Music Analytics</h1>', unsafe_allow_html=True)
#     st.info("📈 Analytics dashboard coming soon!")

# elif page == "📝 Playlists":
#     st.markdown('<h1 class="main-header">📝 My Playlists</h1>', unsafe_allow_html=True)
#     st.info("🎵 Playlist management features coming soon!")

# elif page == "⚙️ Settings":
#     st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
#     st.info("⚙️ Settings panel coming soon!")

# st.markdown('''
# <div style="text-align: center; padding: 1.5rem 0 0.5rem 0; color: #1DB954; font-weight: bold; font-size:1.18rem;">
#     🎵 Spotify Song Recommender | Modern UI | Powered by AI
#     <br>
#     <span style="font-size:0.92rem;color:#222;">Discover • Enjoy • Share</span>
# </div>
# ''', unsafe_allow_html=True)



# import streamlit as st
# import pandas as pd
# import numpy as np
# from numpy import load
# from scipy.sparse import load_npz
# from joblib import load as joblib_load
# from content_based_filtering import content_recommendation
# from collaborative_filtering import collaborative_recommendation
# from hybrid_recommendations import HybridRecommenderSystem
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime, timedelta

# # --------- ENHANCED CSS for modern design ---------
# st.markdown("""
# <style>
#     /* ...YOUR CSS (unchanged, paste all of it from your post)... */
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
#     .stApp { font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
#     /* ...the rest of your style definitions, unchanged... */
# </style>
# """, unsafe_allow_html=True)

# # ------ SESSION STATE ------
# if 'actions' not in st.session_state:
#     st.session_state.actions = {}
# if 'playlists' not in st.session_state:
#     st.session_state.playlists = {}
# if 'recommendation_history' not in st.session_state:
#     st.session_state.recommendation_history = []
# if 'user_preferences' not in st.session_state:
#     st.session_state.user_preferences = {}
# if 'liked_songs' not in st.session_state:
#     st.session_state.liked_songs = set()
# if 'disliked_songs' not in st.session_state:
#     st.session_state.disliked_songs = set()
# if 'queued_songs' not in st.session_state:
#     st.session_state.queued_songs = set()
# if 'selected_mood' not in st.session_state:
#     st.session_state.selected_mood = "Custom"
# if 'current_recommendations' not in st.session_state:
#     st.session_state.current_recommendations = pd.DataFrame()
# if 'playlist_actions' not in st.session_state:
#     st.session_state.playlist_actions = {}

# # -------- CALLBACKS (unchanged) ---------
# def like_callback(song_key):
#     st.session_state.liked_songs.add(song_key)
#     st.session_state.disliked_songs.discard(song_key)
# def dislike_callback(song_key):
#     st.session_state.disliked_songs.add(song_key)
#     st.session_state.liked_songs.discard(song_key)
# def queue_callback(song_key):
#     st.session_state.queued_songs.add(song_key)
# def playlist_callback(song_key, rec):
#     st.session_state.playlist_actions[song_key] = {
#         'name': rec['name'],
#         'artist': rec['artist'],
#         'url': rec.get('spotify_preview_url', '')
#     }
# def quick_add_callback(song_key, selected, rec):
#     song_info = st.session_state.playlist_actions[song_key]
#     if selected != "Select playlist...":
#         exists = any(
#             s['name'] == song_info['name'] and s['artist'] == song_info['artist']
#             for s in st.session_state.playlists[selected]
#         )
#         if not exists:
#             st.session_state.playlists[selected].append(song_info)
#         del st.session_state.playlist_actions[song_key]
# def quick_cancel_callback(song_key):
#     if song_key in st.session_state.playlist_actions:
#         del st.session_state.playlist_actions[song_key]

# # DATA LOADING (unchanged)
# @st.cache_data
# def load_app_data():
#     try:
#         return {
#             'songs_data': pd.read_csv("data/cleaned_data.csv"),
#             'transformed_data': load_npz("data/transformed_data.npz"),
#             'track_ids': load("data/track_ids.npy", allow_pickle=True),
#             'filtered_data': pd.read_csv("data/collab_filtered_data.csv"),
#             'interaction_matrix': load_npz("data/interaction_matrix.npz"),
#             'transformed_hybrid_data': load_npz("data/transformed_hybrid_data.npz"),
#             'transformer': joblib_load('transformer.joblib')
#         }
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         return None

# data = load_app_data()
# if data is None:
#     st.stop()
# songs_data = data['songs_data']
# transformed_data = data['transformed_data']
# track_ids = data['track_ids']
# filtered_data = data['filtered_data']
# interaction_matrix = data['interaction_matrix']
# transformed_hybrid_data = data['transformed_hybrid_data']
# transformer = data['transformer']

# # ---- MOODS (unchanged) ----
# MOOD_PRESETS = {
#     "😊 Happy": {"danceability": 0.8, "energy": 0.75, "valence": 0.9, "tempo": 130.0, "acousticness": 0.2, "instrumentalness": 0.1, "liveness": 0.3, "loudness": -5.0, "key": 2},
#     "😢 Sad": {"danceability": 0.3, "energy": 0.25, "valence": 0.15, "tempo": 75.0, "acousticness": 0.7, "instrumentalness": 0.3, "liveness": 0.1, "loudness": -15.0, "key": 9},
#     "🔥 Energetic": {"danceability": 0.9, "energy": 0.95, "valence": 0.8, "tempo": 150.0, "acousticness": 0.1, "instrumentalness": 0.05, "liveness": 0.4, "loudness": -3.0, "key": 7},
#     "😴 Relaxed": {"danceability": 0.4, "energy": 0.3, "valence": 0.6, "tempo": 85.0, "acousticness": 0.6, "instrumentalness": 0.4, "liveness": 0.15, "loudness": -12.0, "key": 5},
#     "💪 Workout": {"danceability": 0.85, "energy": 0.9, "valence": 0.75, "tempo": 140.0, "acousticness": 0.15, "instrumentalness": 0.1, "liveness": 0.3, "loudness": -4.0, "key": 1},
#     "🎉 Party": {"danceability": 0.95, "energy": 0.85, "valence": 0.85, "tempo": 125.0, "acousticness": 0.1, "instrumentalness": 0.05, "liveness": 0.5, "loudness": -3.5, "key": 4},
#     "🧘 Meditation": {"danceability": 0.2, "energy": 0.15, "valence": 0.65, "tempo": 60.0, "acousticness": 0.8, "instrumentalness": 0.7, "liveness": 0.05, "loudness": -20.0, "key": 3}
# }
# def get_mood_values(selected_mood):
#     if selected_mood == "Custom" or selected_mood not in MOOD_PRESETS:
#         return {"danceability": 0.5, "energy": 0.5, "valence": 0.5, "tempo": 120.0, "acousticness": 0.5, "instrumentalness": 0.5, "liveness": 0.2, "loudness": -10.0, "key": 0}
#     return MOOD_PRESETS[selected_mood]

# def recommend_by_available_features(user_preferences, songs_df, num_recs, exclude_songs=None):
#     try:
#         available_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'tempo', 'loudness', 'key']
#         existing_features = [f for f in available_features if f in songs_df.columns]
#         if not existing_features:
#             return pd.DataFrame()
#         filtered_songs = songs_df.copy()
#         if exclude_songs:
#             exclude_names = [song.split('_')[0] for song in exclude_songs]
#             filtered_songs = filtered_songs[~filtered_songs['name'].isin(exclude_names)]
#         user_vector = np.array([user_preferences.get(feature, 0.5) for feature in existing_features])
#         song_vectors = filtered_songs[existing_features].values
#         song_vectors = np.nan_to_num(song_vectors, nan=0.0)
#         user_vector_2d = user_vector.reshape(1, -1)
#         similarity_scores = cosine_similarity(user_vector_2d, song_vectors)[0]
#         top_indices = np.argsort(similarity_scores)[-num_recs:][::-1]
#         recommendations = filtered_songs.iloc[top_indices][['name', 'artist', 'spotify_preview_url']].copy()
#         recommendations['similarity_score'] = similarity_scores[top_indices]
#         return recommendations.reset_index(drop=True)
#     except Exception as e:
#         st.error(f"Error in recommendation: {e}")
#         return pd.DataFrame()

# # ---- ENHANCED UI Display: Recommendations ----
# def display_static_recommendations(recommendations, show_scores=False):
#     if recommendations.empty:
#         st.warning("🎵 No recommendations found. Try adjusting your preferences!")
#         return

#     st.session_state.current_recommendations = recommendations
#     for ind, rec in recommendations.iterrows():
#         song_name = rec['name'].title()
#         artist_name = rec['artist'].title()
#         song_key = f"{rec['name']}_{rec['artist']}"

#         st.markdown('<div class="song-card">', unsafe_allow_html=True)
#         if ind == 0:
#             st.markdown('<div class="now-playing">🎵 Currently Playing</div>', unsafe_allow_html=True)
#         elif ind == 1:
#             st.markdown('<div class="next-up">🎶 Next Up</div>', unsafe_allow_html=True)
#         st.markdown(f'<div class="song-title">{song_name}</div>', unsafe_allow_html=True)
#         st.markdown(f'<div class="song-artist">by {artist_name}</div>', unsafe_allow_html=True)

#         if show_scores and 'similarity_score' in rec:
#             st.markdown(f'<div class="song-match">✨ Match: {rec["similarity_score"]:.1%}</div>', unsafe_allow_html=True)

#         # Status badges
#         status_html = '<div class="status-badges">'
#         if song_key in st.session_state.liked_songs:
#             status_html += '<span class="status-badge badge-liked">💚 Liked</span>'
#         if song_key in st.session_state.disliked_songs:
#             status_html += '<span class="status-badge badge-disliked">💔 Disliked</span>'
#         if song_key in st.session_state.queued_songs:
#             status_html += '<span class="status-badge badge-queued">✅ Queued</span>'
#         status_html += '</div>'
#         st.markdown(status_html, unsafe_allow_html=True)

#         # BUTTONS: Use a form for each song for best reactivity, no st.rerun!
#         with st.form(key=f"actions_{ind}_{song_key}", clear_on_submit=False):
#             cols = st.columns(4)
#             like = cols[0].form_submit_button("❤️ Like", help="Like this song")
#             dislike = cols[1].form_submit_button("👎 Pass", help="Not for me")
#             queue = cols[2].form_submit_button("➕ Queue", help="Add to queue")
#             playlist = cols[3].form_submit_button("📋 Playlist", help="Add to playlist")

#             if like:
#                 like_callback(song_key)
#             if dislike:
#                 dislike_callback(song_key)
#             if queue:
#                 queue_callback(song_key)
#             if playlist:
#                 playlist_callback(song_key, rec)

#         st.markdown('<div class="audio-preview">', unsafe_allow_html=True)
#         try:
#             if pd.notna(rec['spotify_preview_url']) and rec['spotify_preview_url'].strip():
#                 st.markdown('<div class="audio-available">🔊 Audio Preview Available</div>', unsafe_allow_html=True)
#                 st.audio(rec['spotify_preview_url'])
#             else:
#                 st.markdown('<div class="audio-unavailable">🔇 Audio preview not available</div>', unsafe_allow_html=True)
#         except Exception:
#             st.markdown('<div class="audio-unavailable">🔇 Audio preview not available</div>', unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)
#         st.markdown('<br>', unsafe_allow_html=True)

# # --- Enhanced Sidebar Navigation ---
# st.sidebar.markdown('<h1 style="color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🎵 Navigation</h1>', unsafe_allow_html=True)
# page = st.sidebar.selectbox("Choose Page", [ "🏠 Home", "🔍 Discover", "📊 Analytics", "📝 Playlists", "⚙️ Settings"])

# # --- Main Home Page Logic ---
# if page == "🏠 Home":
#     st.markdown('<h1 class="main-header">🎵 Spotify Song Recommender</h1>', unsafe_allow_html=True)
#     st.markdown('<div class="subtitle">🎧 Discover your next favorite song with AI-powered recommendations</div>', unsafe_allow_html=True)

#     recommender_type = st.selectbox('🤖 Choose Your Algorithm:',
#         ['Feature Sliders', 'Content-Based Filtering', 'Collaborative Filtering', 'Hybrid Recommender System'],
#         index=0, help="Different algorithms provide different types of recommendations")
#     enable_song_inputs = recommender_type != "Feature Sliders"

#     col1, col2 = st.columns(2)
#     with col1:
#         song_name = st.text_input('🎵 Enter a song name:', disabled=not enable_song_inputs, help="Required for content-based and collaborative filtering")
#     with col2:
#         artist_name = st.text_input('🎤 Enter the artist name:', disabled=not enable_song_inputs, help="Required for content-based and collaborative filtering")
#     col1, col2 = st.columns(2)
#     with col1:
#         k = st.selectbox('📊 Number of recommendations:', [5, 10, 15, 20], index=1)
#     with col2:
#         exclude_heard = st.checkbox("🚫 Exclude liked/disliked songs", value=True, help="Exclude songs you've already rated")

#     # Sidebar controls
#     st.sidebar.markdown('<h2 style="color: white; margin-top: 2rem;">🎵 Audio Features</h2>', unsafe_allow_html=True)
#     st.sidebar.markdown('<h3 style="color: white; font-size: 1.1rem;">😊 Quick Mood Presets</h3>', unsafe_allow_html=True)
#     mood_options = ["Custom"] + list(MOOD_PRESETS.keys())
#     new_mood = st.sidebar.selectbox("Choose a mood:", mood_options, key="mood_selector", help="Select a preset mood or customize manually")
#     if new_mood != st.session_state.selected_mood:
#         st.session_state.selected_mood = new_mood
#         if new_mood != "Custom":
#             st.rerun()
#     mood_values = get_mood_values(st.session_state.selected_mood)
#     if st.session_state.selected_mood != "Custom":
#         st.sidebar.markdown(f'<div class="mood-active">🎭 {st.session_state.selected_mood} Active</div>', unsafe_allow_html=True)
#     st.sidebar.markdown('<h3 style="color: white; font-size: 1.1rem; margin-top: 1.5rem;">🎛️ Core Features</h3>', unsafe_allow_html=True)
#     danceability = st.sidebar.slider('💃 Danceability', 0.0, 1.0, float(mood_values["danceability"]), help="How suitable a track is for dancing")
#     energy = st.sidebar.slider('⚡ Energy', 0.0, 1.0, float(mood_values["energy"]), help="Perceptual measure of intensity and power")
#     valence = st.sidebar.slider('😊 Valence', 0.0, 1.0, float(mood_values["valence"]), help="Musical positiveness conveyed by a track")
#     tempo = st.sidebar.slider('🥁 Tempo', 60.0, 200.0, float(mood_values["tempo"]), help="Overall estimated tempo in beats per minute")
#     with st.sidebar.expander("🎛️ Advanced Features", expanded=False):
#         acousticness = st.slider('🎸 Acousticness', 0.0, 1.0, float(mood_values["acousticness"]), help="Confidence measure of whether the track is acoustic")
#         instrumentalness = st.slider('🎹 Instrumentalness', 0.0, 1.0, float(mood_values["instrumentalness"]), help="Predicts whether a track contains no vocals")
#         liveness = st.slider('🎤 Liveness', 0.0, 1.0, float(mood_values["liveness"]), help="Detects the presence of an audience in the recording")
#         loudness = st.sidebar.slider('🔊 Loudness', -60.0, 0.0, float(mood_values["loudness"]), help="Overall loudness of a track in decibels")
#         key = st.selectbox('🎼 Key', list(range(12)), index=int(mood_values["key"]), help="The key the track is in")

#     # Preferences display
#     st.markdown("### 🎯 Current Preferences")
#     pref_col1, pref_col2, pref_col3, pref_col4 = st.columns(4)
#     with pref_col1:
#         st.markdown(f'''
#         <div class="metric-card"><div class="metric-value">{danceability:.2f}</div>
#         <div class="metric-label">💃 Danceability</div></div>''', unsafe_allow_html=True)
#     with pref_col2:
#         st.markdown(f'''
#         <div class="metric-card"><div class="metric-value">{energy:.2f}</div>
#         <div class="metric-label">⚡ Energy</div></div>''', unsafe_allow_html=True)
#     with pref_col3:
#         st.markdown(f'''
#         <div class="metric-card"><div class="metric-value">{valence:.2f}</div>
#         <div class="metric-label">😊 Valence</div></div>''', unsafe_allow_html=True)
#     with pref_col4:
#         st.markdown(f'''
#         <div class="metric-card"><div class="metric-value">{tempo:.0f}</div>
#         <div class="metric-label">🥁 BPM</div></div>''', unsafe_allow_html=True)
#     st.markdown('<br>', unsafe_allow_html=True)

#     # Get recommendations button
#     if st.button('🎵 Get Recommendations', type="primary", use_container_width=True):
#         with st.spinner('🎵 Finding your perfect songs...'):
#             exclude_list = list(st.session_state.liked_songs) + list(st.session_state.disliked_songs) if exclude_heard else []
#             user_preferences = {
#                 'danceability': danceability, 'energy': energy, 'valence': valence,
#                 'acousticness': acousticness, 'instrumentalness': instrumentalness,
#                 'liveness': liveness, 'tempo': tempo, 'loudness': loudness, 'key': key
#             }
#             if recommender_type == 'Feature Sliders':
#                 recommendations = recommend_by_available_features(user_preferences, songs_data, k, exclude_list)
#                 if not recommendations.empty:
#                     st.success(f'🎵 Found {len(recommendations)} amazing songs for you!')
#                     display_static_recommendations(recommendations, show_scores=True)
#                 else:
#                     st.warning("🎵 No songs found matching your criteria. Try adjusting your preferences!")
#             else:
#                 if not song_name or not artist_name:
#                     st.warning("⚠️ Please enter BOTH song name and artist name for this algorithm.")
#                 else:
#                     song_name_low = song_name.lower().strip()
#                     artist_name_low = artist_name.lower().strip()
#                     if recommender_type == 'Content-Based Filtering':
#                         if ((songs_data["name"] == song_name_low) & (songs_data['artist'] == artist_name_low)).any():
#                             recommendations = content_recommendation(song_name_low, artist_name_low, songs_data, transformed_data, k)
#                             st.success(f'🎵 Found songs similar to **{song_name}** by **{artist_name}**!')
#                             display_static_recommendations(recommendations)
#                         else:
#                             st.error("❌ Song not found in database. Please check the spelling and try again.")
#                     elif recommender_type == 'Collaborative Filtering':
#                         if ((filtered_data["name"] == song_name_low) & (filtered_data['artist'] == artist_name_low)).any():
#                             recommendations = collaborative_recommendation(song_name_low, artist_name_low, track_ids, filtered_data, interaction_matrix, k)
#                             st.success(f'🎵 Collaborative recommendations for **{song_name}** by **{artist_name}**!')
#                             display_static_recommendations(recommendations)
#                         else:
#                             st.error("❌ Song not found in collaborative dataset. Try a different song.")
#                     elif recommender_type == 'Hybrid Recommender System':
#                         if ((filtered_data["name"] == song_name_low) & (filtered_data['artist'] == artist_name_low)).any():
#                             diversity = st.slider("🔀 Diversity Level", 1, 9, 5, help="Higher values = more diverse recommendations")
#                             content_based_weight = 1 - (diversity / 10)
#                             recommender = HybridRecommenderSystem(
#                                 number_of_recommendations=k, weight_content_based=content_based_weight
#                             )
#                             recommendations = recommender.give_recommendations(
#                                 song_name=song_name_low,
#                                 artist_name=artist_name_low,
#                                 songs_data=filtered_data,
#                                 track_ids=track_ids,
#                                 transformed_matrix=transformed_hybrid_data,
#                                 interaction_matrix=interaction_matrix
#                             )
#                             st.success(f'🎵 Hybrid recommendations for **{song_name}** by **{artist_name}**!')
#                             display_static_recommendations(recommendations)
#                         else:
#                             st.error("❌ Song not found in hybrid dataset. Try a different song.")
#     elif not st.session_state.current_recommendations.empty:
#         st.markdown("### 🎵 Your Last Recommendations")
#         display_static_recommendations(st.session_state.current_recommendations, show_scores=True)

# elif page == "🔍 Discover":
#     st.markdown('<h1 class="main-header">🔍 Advanced Discovery</h1>', unsafe_allow_html=True)
#     st.markdown('<div class="subtitle">Explore new music territories</div>', unsafe_allow_html=True)
#     st.info("🚧 Advanced discovery features coming soon! This will include genre exploration, trending tracks, and personalized discovery feeds.")

# elif page == "📊 Analytics":
#     st.markdown('<h1 class="main-header">📊 Your Music Analytics</h1>', unsafe_allow_html=True)
#     st.markdown('<div class="subtitle">Insights into your music taste</div>', unsafe_allow_html=True)
#     st.info("📈 Analytics dashboard coming soon! Track your listening patterns, favorite genres, and recommendation accuracy.")

# elif page == "📝 Playlists":
#     st.markdown('<h1 class="main-header">📝 My Playlists</h1>', unsafe_allow_html=True)
#     st.markdown('<div class="subtitle">Manage your curated collections</div>', unsafe_allow_html=True)
#     st.info("🎵 Playlist management features coming soon! Create, edit, and share your custom playlists.")

# elif page == "⚙️ Settings":
#     st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
#     st.markdown('<div class="subtitle">Customize your experience</div>', unsafe_allow_html=True)
#     st.info("⚙️ Settings panel coming soon! Adjust preferences, themes, and recommendation parameters.")

# # ---- Enhanced Footer ----
# st.markdown('''
# <div class="footer">
#     🎵 Spotify Song Recommender | Enhanced UI | Powered by AI
#     <br>
#     <small>Discover • Enjoy • Share</small>
# </div>
# ''', unsafe_allow_html=True)




import streamlit as st
import pandas as pd
import numpy as np
from numpy import load
from scipy.sparse import load_npz
from joblib import load as joblib_load
from content_based_filtering import content_recommendation
from collaborative_filtering import collaborative_recommendation
from hybrid_recommendations import HybridRecommenderSystem
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

# --------- ENHANCED CSS with Spotify-style BUTTONS and UI ---------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh; }
    .main .block-container {
        padding-top: 2rem; padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 1rem;
    }
    .main-header {
        font-size: 3.5rem; font-weight: 700; text-align: center;
        background: linear-gradient(135deg, #1DB954, #1ed760, #00d4aa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.5rem;
        text-shadow: 0 4px 8px rgba(29, 185, 84, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow { from { filter: drop-shadow(0 0 5px rgba(29, 185, 84, 0.5)); } to { filter: drop-shadow(0 0 20px rgba(29, 185, 84, 0.8)); } }
    .subtitle { text-align: center; font-size: 1.2rem; color: #666;
        margin-bottom: 2rem; font-weight: 400; }
    .css-1d391kg {
        background: linear-gradient(180deg, #1DB954 0%, #1ed760 100%);
        border-radius: 0 20px 20px 0; }
    .sidebar .sidebar-content { background: transparent; }
    .sidebar h1, .sidebar h2, .sidebar h3 { color: white !important; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3); }
    .sidebar .stSelectbox label, .sidebar .stSlider label { color: white !important; font-weight: 500; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem; border-radius: 15px; color: white; text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease; border: 1px solid rgba(255,255,255,0.2);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4); }
    .metric-value { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
    .metric-label { font-size: 0.9rem; opacity: 0.9; font-weight: 500; }
    .song-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem; border-radius: 15px; margin: 1rem 0;
        border-left: 5px solid #1DB954;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1); transition: all 0.3s ease;
        position: relative; overflow: hidden;
    }
    .song-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #1DB954, #1ed760, #00d4aa); }
    .song-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
        border-left-color: #1ed760; }
    .song-title { font-size: 1.3rem; font-weight: 600; color: #1a202c;
        margin-bottom: 0.3rem; }
    .song-artist { font-size: 1rem; color: #4a5568; margin-bottom: 0.5rem; font-weight: 500; }
    .song-match { font-size: 0.9rem; color: #1DB954; font-weight: 600; margin-bottom: 1rem; }
    .action-buttons { display: flex; gap: 0.75rem; margin: 1rem 0; flex-wrap: wrap; }
    .action-btn {
        padding: 0.5rem 1rem; border-radius: 25px; border: 2px solid transparent; font-weight: 500; font-size: 0.9rem; cursor: pointer; transition: all 0.3s ease; display: inline-flex; align-items: center; gap: 0.5rem; text-decoration: none; min-width: 80px; justify-content: center;
    }
    /* Per-button color classes: optional. All main/primary buttons below. */
    /* Primary Streamlit action buttons (Get Recommendations etc) */
    .stButton > button,
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #1DB954, #1ed760, #00d4aa) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.08rem !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 15px rgba(29,185,84,0.23) !important;
    }
    .stButton > button:hover, .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1ed760, #00d4aa) !important;
        color: #232323 !important;
    }
    /* Like, Pass, ... card actions: keep color as you like, no global red */
    .btn-like, .btn-dislike, .btn-queue, .btn-playlist { filter: none !important; }
    /* Status badges etc (unchanged) */
    .status-badges { display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }
    .status-badge { padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; display: inline-flex; align-items: center; gap: 0.25rem; }
    .badge-liked { background: rgba(255, 107, 107, 0.1); color: #ff6b6b; border: 1px solid rgba(255, 107, 107, 0.3); }
    .badge-disliked { background: rgba(116, 185, 255, 0.1); color: #74b9ff; border: 1px solid rgba(116, 185, 255, 0.3); }
    .badge-queued { background: rgba(162, 155, 254, 0.1); color: #a29bfe; border: 1px solid rgba(162, 155, 254, 0.3); }
    .now-playing {
        background: linear-gradient(135deg, #1DB954, #1ed760);
        color: white; padding: 0.5rem 1rem; border-radius: 25px;
        font-size: 0.9rem; font-weight: 600; margin-bottom: 1rem;
        display: inline-flex; align-items: center; gap: 0.5rem; animation: pulse 2s infinite;
    }
    .next-up { background: rgba(29, 185, 84, 0.1); color: #1DB954;
        padding: 0.5rem 1rem; border-radius: 25px; font-size: 0.9rem;
        font-weight: 600; margin-bottom: 1rem;
        display: inline-flex; align-items: center; gap: 0.5rem; border: 1px solid rgba(29, 185, 84, 0.3); }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(29, 185, 84, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(29, 185, 84, 0); } 100% { box-shadow: 0 0 0 0 rgba(29, 185, 84, 0); } }
    .audio-preview { background: linear-gradient(135deg, #f8fafc, #e2e8f0); padding: 1rem; border-radius: 10px; margin-top: 1rem; border: 1px solid rgba(29, 185, 84, 0.2);}
    .audio-available { color: #1DB954; font-weight: 500; display: flex; align-items: center; gap: 0.5rem; }
    .audio-unavailable { color: #9ca3af; font-style: italic; display: flex; align-items: center; gap: 0.5rem; }
    .mood-active {
        background: linear-gradient(135deg, #1DB954, #1ed760) !important;
        color: white !important;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        margin: 0.5rem 0;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3);
        animation: glow-mood 2s ease-in-out infinite alternate;
    }
    @keyframes glow-mood { from { box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3);} to { box-shadow: 0 4px 25px rgba(29, 185, 84, 0.5);} }
    .footer { text-align: center; padding: 2rem; margin-top: 3rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border-radius: 15px; font-weight: 500;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);}
    @media (max-width: 768px) {
        .main-header { font-size: 2.5rem; }
        .action-buttons { justify-content: center; }
        .action-btn { min-width: 70px; font-size: 0.8rem; padding: 0.4rem 0.8rem;}
    }
    /* Custom Streamlit overrides for controls */
    .stSelectbox > div > div { border-radius: 10px; border: 2px solid rgba(29, 185, 84, 0.3);}
    .stTextInput > div > div > input { border-radius: 10px; border: 2px solid rgba(29, 185, 84, 0.3);}
    .stSlider > div > div > div { background: linear-gradient(135deg, #1DB954, #1ed760);}
</style>
""", unsafe_allow_html=True)

# ------ SESSION STATE ------
if 'actions' not in st.session_state:
    st.session_state.actions = {}
if 'playlists' not in st.session_state:
    st.session_state.playlists = {}
if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {}
if 'liked_songs' not in st.session_state:
    st.session_state.liked_songs = set()
if 'disliked_songs' not in st.session_state:
    st.session_state.disliked_songs = set()
if 'queued_songs' not in st.session_state:
    st.session_state.queued_songs = set()
if 'selected_mood' not in st.session_state:
    st.session_state.selected_mood = "Custom"
if 'current_recommendations' not in st.session_state:
    st.session_state.current_recommendations = pd.DataFrame()
if 'playlist_actions' not in st.session_state:
    st.session_state.playlist_actions = {}

# --- CALLBACKS (unchanged) ---
def like_callback(song_key):
    st.session_state.liked_songs.add(song_key)
    st.session_state.disliked_songs.discard(song_key)
def dislike_callback(song_key):
    st.session_state.disliked_songs.add(song_key)
    st.session_state.liked_songs.discard(song_key)
def queue_callback(song_key):
    st.session_state.queued_songs.add(song_key)
def playlist_callback(song_key, rec):
    st.session_state.playlist_actions[song_key] = {
        'name': rec['name'],
        'artist': rec['artist'],
        'url': rec.get('spotify_preview_url', '')
    }
def quick_add_callback(song_key, selected, rec):
    song_info = st.session_state.playlist_actions[song_key]
    if selected != "Select playlist...":
        exists = any(
            s['name'] == song_info['name'] and s['artist'] == song_info['artist']
            for s in st.session_state.playlists[selected]
        )
        if not exists:
            st.session_state.playlists[selected].append(song_info)
        del st.session_state.playlist_actions[song_key]
def quick_cancel_callback(song_key):
    if song_key in st.session_state.playlist_actions:
        del st.session_state.playlist_actions[song_key]

# DATA LOADING (unchanged)
@st.cache_data
def load_app_data():
    try:
        return {
            'songs_data': pd.read_csv("data/cleaned_data.csv"),
            'transformed_data': load_npz("data/transformed_data.npz"),
            'track_ids': load("data/track_ids.npy", allow_pickle=True),
            'filtered_data': pd.read_csv("data/collab_filtered_data.csv"),
            'interaction_matrix': load_npz("data/interaction_matrix.npz"),
            'transformed_hybrid_data': load_npz("data/transformed_hybrid_data.npz"),
            'transformer': joblib_load('transformer.joblib')
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_app_data()
if data is None: st.stop()
songs_data = data['songs_data']
transformed_data = data['transformed_data']
track_ids = data['track_ids']
filtered_data = data['filtered_data']
interaction_matrix = data['interaction_matrix']
transformed_hybrid_data = data['transformed_hybrid_data']
transformer = data['transformer']

# ---- MOODS (unchanged) ----
MOOD_PRESETS = {
    "😊 Happy": {...}, # (no change, copy keys/values as before)
    "😢 Sad": {...},
    "🔥 Energetic": {...},
    "😴 Relaxed": {...},
    "💪 Workout": {...},
    "🎉 Party": {...},
    "🧘 Meditation": {...}
}
def get_mood_values(selected_mood):
    if selected_mood == "Custom" or selected_mood not in MOOD_PRESETS:
        return {"danceability": 0.5, "energy": 0.5, "valence": 0.5, "tempo": 120.0, "acousticness": 0.5, "instrumentalness": 0.5, "liveness": 0.2, "loudness": -10.0, "key": 0}
    return MOOD_PRESETS[selected_mood]

def recommend_by_available_features(user_preferences, songs_df, num_recs, exclude_songs=None):
    try:
        available_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'tempo', 'loudness', 'key']
        existing_features = [f for f in available_features if f in songs_df.columns]
        if not existing_features:
            return pd.DataFrame()
        filtered_songs = songs_df.copy()
        if exclude_songs:
            exclude_names = [song.split('_')[0] for song in exclude_songs]
            filtered_songs = filtered_songs[~filtered_songs['name'].isin(exclude_names)]
        user_vector = np.array([user_preferences.get(feature, 0.5) for feature in existing_features])
        song_vectors = filtered_songs[existing_features].values
        song_vectors = np.nan_to_num(song_vectors, nan=0.0)
        user_vector_2d = user_vector.reshape(1, -1)
        similarity_scores = cosine_similarity(user_vector_2d, song_vectors)[0]
        top_indices = np.argsort(similarity_scores)[-num_recs:][::-1]
        recommendations = filtered_songs.iloc[top_indices][['name', 'artist', 'spotify_preview_url']].copy()
        recommendations['similarity_score'] = similarity_scores[top_indices]
        return recommendations.reset_index(drop=True)
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return pd.DataFrame()

# ---- ENHANCED UI Display: Recommendations with modern styling ----
def display_static_recommendations(recommendations, show_scores=False):
    if recommendations.empty:
        st.warning("🎵 No recommendations found. Try adjusting your preferences!")
        return
    st.session_state.current_recommendations = recommendations
    for ind, rec in recommendations.iterrows():
        song_name = rec['name'].title()
        artist_name = rec['artist'].title()
        song_key = f"{rec['name']}_{rec['artist']}"
        st.markdown('<div class="song-card">', unsafe_allow_html=True)
        if ind == 0:
            st.markdown('<div class="now-playing">🎵 Currently Playing</div>', unsafe_allow_html=True)
        elif ind == 1:
            st.markdown('<div class="next-up">🎶 Next Up</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="song-title">{song_name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="song-artist">by {artist_name}</div>', unsafe_allow_html=True)
        if show_scores and 'similarity_score' in rec:
            st.markdown(f'<div class="song-match">✨ Match: {rec["similarity_score"]:.1%}</div>', unsafe_allow_html=True)
        status_html = '<div class="status-badges">'
        if song_key in st.session_state.liked_songs:
            status_html += '<span class="status-badge badge-liked">💚 Liked</span>'
        if song_key in st.session_state.disliked_songs:
            status_html += '<span class="status-badge badge-disliked">💔 Disliked</span>'
        if song_key in st.session_state.queued_songs:
            status_html += '<span class="status-badge badge-queued">✅ Queued</span>'
        status_html += '</div>'
        st.markdown(status_html, unsafe_allow_html=True)
        # Button actions – use form, so UI is "reactive" and fast
        with st.form(key=f"actions_{ind}_{song_key}", clear_on_submit=False):
            cols = st.columns(4)
            like = cols[0].form_submit_button("❤️ Like", help="Like this song")
            dislike = cols[1].form_submit_button("👎 Pass", help="Not for me")
            queue = cols[2].form_submit_button("➕ Queue", help="Add to queue")
            playlist = cols[3].form_submit_button("📋 Playlist", help="Add to playlist")
            if like: like_callback(song_key)
            if dislike: dislike_callback(song_key)
            if queue: queue_callback(song_key)
            if playlist: playlist_callback(song_key, rec)
        # Audio Preview
        st.markdown('<div class="audio-preview">', unsafe_allow_html=True)
        try:
            if pd.notna(rec['spotify_preview_url']) and rec['spotify_preview_url'].strip():
                st.markdown('<div class="audio-available">🔊 Audio Preview Available</div>', unsafe_allow_html=True)
                st.audio(rec['spotify_preview_url'])
            else:
                st.markdown('<div class="audio-unavailable">🔇 Audio preview not available</div>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<div class="audio-unavailable">🔇 Audio preview not available</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)

# --- Enhanced Sidebar Navigation ---
st.sidebar.markdown('<h1 style="color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🎵 Navigation</h1>', unsafe_allow_html=True)
page = st.sidebar.selectbox("Choose Page", ["🏠 Home", "🔍 Discover", "📊 Analytics", "📝 Playlists", "⚙️ Settings"])

# --- Main Home Page Logic ---
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🎵 Spotify Song Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">🎧 Discover your next favorite song with AI-powered recommendations</div>', unsafe_allow_html=True)
    recommender_type = st.selectbox('🤖 Choose Your Algorithm:',
        ['Feature Sliders', 'Content-Based Filtering', 'Collaborative Filtering', 'Hybrid Recommender System'], index=0, help="Different algorithms provide different types of recommendations")
    enable_song_inputs = recommender_type != "Feature Sliders"
    col1, col2 = st.columns(2)
    with col1:
        song_name = st.text_input('🎵 Enter a song name:', disabled=not enable_song_inputs, help="Required for content-based and collaborative filtering")
    with col2:
        artist_name = st.text_input('🎤 Enter the artist name:', disabled=not enable_song_inputs, help="Required for content-based and collaborative filtering")
    col1, col2 = st.columns(2)
    with col1:
        k = st.selectbox('📊 Number of recommendations:', [5, 10, 15, 20], index=1)
    with col2:
        exclude_heard = st.checkbox("🚫 Exclude liked/disliked songs", value=True, help="Exclude songs you've already rated")
    # Sidebar controls
    st.sidebar.markdown('<h2 style="color: white; margin-top: 2rem;">🎵 Audio Features</h2>', unsafe_allow_html=True)
    st.sidebar.markdown('<h3 style="color: white; font-size: 1.1rem;">😊 Quick Mood Presets</h3>', unsafe_allow_html=True)
    mood_options = ["Custom"] + list(MOOD_PRESETS.keys())
    new_mood = st.sidebar.selectbox("Choose a mood:", mood_options, key="mood_selector", help="Select a preset mood or customize manually")
    if new_mood != st.session_state.selected_mood:
        st.session_state.selected_mood = new_mood
        if new_mood != "Custom":
            st.rerun()
    mood_values = get_mood_values(st.session_state.selected_mood)
    if st.session_state.selected_mood != "Custom":
        st.sidebar.markdown(f'<div class="mood-active">🎭 {st.session_state.selected_mood} Active</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<h3 style="color: white; font-size: 1.1rem; margin-top: 1.5rem;">🎛️ Core Features</h3>', unsafe_allow_html=True)
    danceability = st.sidebar.slider('💃 Danceability', 0.0, 1.0, float(mood_values["danceability"]), help="How suitable a track is for dancing")
    energy = st.sidebar.slider('⚡ Energy', 0.0, 1.0, float(mood_values["energy"]), help="Perceptual measure of intensity and power")
    valence = st.sidebar.slider('😊 Valence', 0.0, 1.0, float(mood_values["valence"]), help="Musical positiveness conveyed by a track")
    tempo = st.sidebar.slider('🥁 Tempo', 60.0, 200.0, float(mood_values["tempo"]), help="Overall estimated tempo in beats per minute")
    with st.sidebar.expander("🎛️ Advanced Features", expanded=False):
        acousticness = st.slider('🎸 Acousticness', 0.0, 1.0, float(mood_values["acousticness"]), help="Confidence measure of whether the track is acoustic")
        instrumentalness = st.slider('🎹 Instrumentalness', 0.0, 1.0, float(mood_values["instrumentalness"]), help="Predicts whether a track contains no vocals")
        liveness = st.slider('🎤 Liveness', 0.0, 1.0, float(mood_values["liveness"]), help="Detects the presence of an audience in the recording")
        loudness = st.sidebar.slider('🔊 Loudness', -60.0, 0.0, float(mood_values["loudness"]), help="Overall loudness of a track in decibels")
        key = st.selectbox('🎼 Key', list(range(12)), index=int(mood_values["key"]), help="The key the track is in")
    st.markdown("### 🎯 Current Preferences")
    pref_col1, pref_col2, pref_col3, pref_col4 = st.columns(4)
    with pref_col1:
        st.markdown(f'''<div class="metric-card"><div class="metric-value">{danceability:.2f}</div>
        <div class="metric-label">💃 Danceability</div></div>''', unsafe_allow_html=True)
    with pref_col2:
        st.markdown(f'''<div class="metric-card"><div class="metric-value">{energy:.2f}</div>
        <div class="metric-label">⚡ Energy</div></div>''', unsafe_allow_html=True)
    with pref_col3:
        st.markdown(f'''<div class="metric-card"><div class="metric-value">{valence:.2f}</div>
        <div class="metric-label">😊 Valence</div></div>''', unsafe_allow_html=True)
    with pref_col4:
        st.markdown(f'''<div class="metric-card"><div class="metric-value">{tempo:.0f}</div>
        <div class="metric-label">🥁 BPM</div></div>''', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    if st.button('🎵 Get Recommendations', type="primary", use_container_width=True):
        with st.spinner('🎵 Finding your perfect songs...'):
            exclude_list = list(st.session_state.liked_songs) + list(st.session_state.disliked_songs) if exclude_heard else []
            user_preferences = {
                'danceability': danceability, 'energy': energy, 'valence': valence,
                'acousticness': acousticness, 'instrumentalness': instrumentalness,
                'liveness': liveness, 'tempo': tempo, 'loudness': loudness, 'key': key }
            if recommender_type == 'Feature Sliders':
                recommendations = recommend_by_available_features(user_preferences, songs_data, k, exclude_list)
                if not recommendations.empty:
                    st.success(f'🎵 Found {len(recommendations)} amazing songs for you!')
                    display_static_recommendations(recommendations, show_scores=True)
                else:
                    st.warning("🎵 No songs found matching your criteria. Try adjusting your preferences!")
            else:
                if not song_name or not artist_name:
                    st.warning("⚠️ Please enter BOTH song name and artist name for this algorithm.")
                else:
                    song_name_low = song_name.lower().strip()
                    artist_name_low = artist_name.lower().strip()
                    if recommender_type == 'Content-Based Filtering':
                        if ((songs_data["name"] == song_name_low) & (songs_data['artist'] == artist_name_low)).any():
                            recommendations = content_recommendation(song_name_low, artist_name_low, songs_data, transformed_data, k)
                            st.success(f'🎵 Found songs similar to **{song_name}** by **{artist_name}**!')
                            display_static_recommendations(recommendations)
                        else:
                            st.error("❌ Song not found in database. Please check the spelling and try again.")
                    elif recommender_type == 'Collaborative Filtering':
                        if ((filtered_data["name"] == song_name_low) & (filtered_data['artist'] == artist_name_low)).any():
                            recommendations = collaborative_recommendation(song_name_low, artist_name_low, track_ids, filtered_data, interaction_matrix, k)
                            st.success(f'🎵 Collaborative recommendations for **{song_name}** by **{artist_name}**!')
                            display_static_recommendations(recommendations)
                        else:
                            st.error("❌ Song not found in collaborative dataset. Try a different song.")
                    elif recommender_type == 'Hybrid Recommender System':
                        if ((filtered_data["name"] == song_name_low) & (filtered_data['artist'] == artist_name_low)).any():
                            diversity = st.slider("🔀 Diversity Level", 1, 9, 5, help="Higher values = more diverse recommendations")
                            content_based_weight = 1 - (diversity / 10)
                            recommender = HybridRecommenderSystem(
                                number_of_recommendations=k, weight_content_based=content_based_weight)
                            recommendations = recommender.give_recommendations(
                                song_name=song_name_low,
                                artist_name=artist_name_low,
                                songs_data=filtered_data,
                                track_ids=track_ids,
                                transformed_matrix=transformed_hybrid_data,
                                interaction_matrix=interaction_matrix
                            )
                            st.success(f'🎵 Hybrid recommendations for **{song_name}** by **{artist_name}**!')
                            display_static_recommendations(recommendations)
                        else:
                            st.error("❌ Song not found in hybrid dataset. Try a different song.")
    elif not st.session_state.current_recommendations.empty:
        st.markdown("### 🎵 Your Last Recommendations")
        display_static_recommendations(st.session_state.current_recommendations, show_scores=True)
elif page == "🔍 Discover":
    st.markdown('<h1 class="main-header">🔍 Advanced Discovery</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Explore new music territories</div>', unsafe_allow_html=True)
    st.info("🚧 Advanced discovery features coming soon! This will include genre exploration, trending tracks, and personalized discovery feeds.")
elif page == "📊 Analytics":
    st.markdown('<h1 class="main-header">📊 Your Music Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Insights into your music taste</div>', unsafe_allow_html=True)
    st.info("📈 Analytics dashboard coming soon! Track your listening patterns, favorite genres, and recommendation accuracy.")
elif page == "📝 Playlists":
    st.markdown('<h1 class="main-header">📝 My Playlists</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Manage your curated collections</div>', unsafe_allow_html=True)
    st.info("🎵 Playlist management features coming soon! Create, edit, and share your custom playlists.")
elif page == "⚙️ Settings":
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Customize your experience</div>', unsafe_allow_html=True)
    st.info("⚙️ Settings panel coming soon! Adjust preferences, themes, and recommendation parameters.")

st.markdown('''
<div class="footer">
    🎵 Spotify Song Recommender | Enhanced UI | Powered by AI
    <br>
    <small>Discover • Enjoy • Share</small>
</div>
''', unsafe_allow_html=True)


