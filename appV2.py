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
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="🎵 Spotify Song Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .song-container {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1DB954;
    }
    .action-status {
        background: #e8f5e8;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        display: inline-block;
        margin: 2px;
    }
    .liked { background: #d4edda; color: #155724; }
    .disliked { background: #f8d7da; color: #721c24; }
    .queued { background: #d1ecf1; color: #0c5460; }
</style>
""", unsafe_allow_html=True)

# Initialize session state - NEVER MODIFY THESE DURING RECOMMENDATIONS
if 'user_actions' not in st.session_state:
    st.session_state.user_actions = {
        'liked': set(),
        'disliked': set(), 
        'queued': set(),
        'playlists': {},
        'pending_playlist_adds': {}
    }

if 'app_data' not in st.session_state:
    st.session_state.app_data = None

if 'current_recs' not in st.session_state:
    st.session_state.current_recs = pd.DataFrame()

# Load data ONCE
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

# Load data once
if st.session_state.app_data is None:
    st.session_state.app_data = load_app_data()

if st.session_state.app_data is None:
    st.stop()

data = st.session_state.app_data
songs_data = data['songs_data']
transformed_data = data['transformed_data']
track_ids = data['track_ids']
filtered_data = data['filtered_data']
interaction_matrix = data['interaction_matrix']
transformed_hybrid_data = data['transformed_hybrid_data']
transformer = data['transformer']

# Mood presets
MOOD_PRESETS = {
    "😊 Happy": {"danceability": 0.8, "energy": 0.75, "valence": 0.9, "tempo": 130.0, "acousticness": 0.2, "instrumentalness": 0.1, "liveness": 0.3, "loudness": -5.0, "key": 2},
    "😢 Sad": {"danceability": 0.3, "energy": 0.25, "valence": 0.15, "tempo": 75.0, "acousticness": 0.7, "instrumentalness": 0.3, "liveness": 0.1, "loudness": -15.0, "key": 9},
    "🔥 Energetic": {"danceability": 0.9, "energy": 0.95, "valence": 0.8, "tempo": 150.0, "acousticness": 0.1, "instrumentalness": 0.05, "liveness": 0.4, "loudness": -3.0, "key": 7},
    "😴 Relaxed": {"danceability": 0.4, "energy": 0.3, "valence": 0.6, "tempo": 85.0, "acousticness": 0.6, "instrumentalness": 0.4, "liveness": 0.15, "loudness": -12.0, "key": 5},
    "💪 Workout": {"danceability": 0.85, "energy": 0.9, "valence": 0.75, "tempo": 140.0, "acousticness": 0.15, "instrumentalness": 0.1, "liveness": 0.3, "loudness": -4.0, "key": 1}
}

# Navigation
st.sidebar.markdown('<h1 style="color: #1DB954;">🎵 Navigation</h1>', unsafe_allow_html=True)
page = st.sidebar.selectbox("Choose Page", ["🏠 Home", "📝 Playlists", "📊 Analytics", "⚙️ Settings"])

# Recommendation function
def recommend_by_features(user_preferences, songs_df, num_recs, exclude_songs=None):
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
        st.error(f"Error: {e}")
        return pd.DataFrame()

# FRAGMENT: Action buttons that NEVER cause page refresh
@st.fragment
def action_buttons(song_key, song_name, artist_name):
    """Fragment for action buttons - isolated from main app state"""
    
    # Get current status
    is_liked = song_key in st.session_state.user_actions['liked']
    is_disliked = song_key in st.session_state.user_actions['disliked']
    is_queued = song_key in st.session_state.user_actions['queued']
    
    # Action buttons in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        like_emoji = "💚" if is_liked else "❤️"
        if st.button(like_emoji, key=f"like_{song_key}", help="Like this song"):
            st.session_state.user_actions['liked'].add(song_key)
            st.session_state.user_actions['disliked'].discard(song_key)
            st.write("✅ Liked!")
    
    with col2:
        dislike_emoji = "💔" if is_disliked else "👎"
        if st.button(dislike_emoji, key=f"dislike_{song_key}", help="Not for me"):
            st.session_state.user_actions['disliked'].add(song_key)
            st.session_state.user_actions['liked'].discard(song_key)
            st.write("✅ Noted!")
    
    with col3:
        queue_emoji = "✅" if is_queued else "➕"
        if st.button(queue_emoji, key=f"queue_{song_key}", help="Add to queue"):
            st.session_state.user_actions['queued'].add(song_key)
            st.write("✅ Queued!")
    
    with col4:
        if st.button("📋", key=f"playlist_{song_key}", help="Add to playlist"):
            st.session_state.user_actions['pending_playlist_adds'][song_key] = {
                'name': song_name, 'artist': artist_name
            }
            st.write("📋 Ready!")

# FRAGMENT: Playlist selector that doesn't refresh main page
@st.fragment 
def playlist_selector(song_key):
    """Fragment for playlist selection - isolated"""
    if song_key in st.session_state.user_actions['pending_playlist_adds']:
        song_info = st.session_state.user_actions['pending_playlist_adds'][song_key]
        
        if st.session_state.user_actions['playlists']:
            playlist_options = ["Select playlist..."] + list(st.session_state.user_actions['playlists'].keys())
            selected = st.selectbox(
                f"Add '{song_info['name']}' to:", 
                playlist_options,
                key=f"select_{song_key}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Add", key=f"add_{song_key}") and selected != "Select playlist...":
                    # Check if already exists
                    existing = any(
                        s['name'] == song_info['name']
                        for s in st.session_state.user_actions['playlists'][selected]
                    )
                    
                    if not existing:
                        st.session_state.user_actions['playlists'][selected].append(song_info)
                        st.write(f"✅ Added to {selected}!")
                    else:
                        st.write("Already in playlist!")
                    
                    del st.session_state.user_actions['pending_playlist_adds'][song_key]
            
            with col2:
                if st.button("❌ Cancel", key=f"cancel_{song_key}"):
                    del st.session_state.user_actions['pending_playlist_adds'][song_key]
        else:
            st.info("Create a playlist first!")
            if st.button("❌ Cancel", key=f"cancel_no_playlist_{song_key}"):
                del st.session_state.user_actions['pending_playlist_adds'][song_key]

# STATIC recommendations display function
def display_recommendations(recommendations, show_scores=False):
    """Display recommendations with fragmented action buttons"""
    if recommendations.empty:
        st.warning("No recommendations found.")
        return
    
    for ind, rec in recommendations.iterrows():
        song_name = rec['name'].title()
        artist_name = rec['artist'].title()
        song_key = f"{rec['name']}_{rec['artist']}"
        
        # Song container
        with st.container():
            st.markdown('<div class="song-container">', unsafe_allow_html=True)
            
            # Song info
            col1, col2 = st.columns([3, 1])
            with col1:
                if ind == 0:
                    st.markdown("## 🎵 Currently Playing")
                elif ind == 1:
                    st.markdown("### 🎶 Next Up")
                
                st.markdown(f"**{song_name}** by **{artist_name}**")
                
                if show_scores and 'similarity_score' in rec:
                    st.write(f"*Match: {rec['similarity_score']:.1%}*")
            
            with col2:
                # Status indicators
                status_html = ""
                if song_key in st.session_state.user_actions['liked']:
                    status_html += '<span class="action-status liked">💚 Liked</span>'
                if song_key in st.session_state.user_actions['disliked']:
                    status_html += '<span class="action-status disliked">💔 Disliked</span>'
                if song_key in st.session_state.user_actions['queued']:
                    status_html += '<span class="action-status queued">✅ Queued</span>'
                
                if status_html:
                    st.markdown(status_html, unsafe_allow_html=True)
            
            # Action buttons (FRAGMENTED - NO REFRESH)
            action_buttons(song_key, rec['name'], rec['artist'])
            
            # Playlist selector (FRAGMENTED - NO REFRESH)
            playlist_selector(song_key)
            
            # Audio player
            try:
                if pd.notna(rec['spotify_preview_url']) and rec['spotify_preview_url'].strip():
                    st.audio(rec['spotify_preview_url'])
                else:
                    st.write("🎵 *Audio preview not available*")
            except:
                st.write("🎵 *Audio preview not available*")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.write('---')

# HOME PAGE
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🎵 Spotify Song Recommender</h1>', unsafe_allow_html=True)
    
    # Stats display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("❤️ Liked", len(st.session_state.user_actions['liked']))
    with col2:
        st.metric("👎 Disliked", len(st.session_state.user_actions['disliked']))
    with col3:
        st.metric("🎵 Queued", len(st.session_state.user_actions['queued']))
    
    # Input section
    col1, col2 = st.columns(2)
    with col1:
        song_name = st.text_input('🎵 Song name:')
    with col2:
        artist_name = st.text_input('🎤 Artist name:')
    
    # Settings
    col1, col2, col3 = st.columns(3)
    with col1:
        k = st.selectbox('📊 Recommendations:', [5, 10, 15, 20], index=1)
    with col2:
        recommender_type = st.selectbox('🤖 Algorithm:', [
            'Feature Sliders', 'Content-Based Filtering', 'Collaborative Filtering', 'Hybrid Recommender System'
        ])
    with col3:
        exclude_heard = st.checkbox("🚫 Exclude liked/disliked", value=True)
    
    # Sidebar features
    st.sidebar.header("🎵 Audio Features")
    
    # Mood presets
    mood_options = ["Custom"] + list(MOOD_PRESETS.keys())
    selected_mood = st.sidebar.selectbox("😊 Mood:", mood_options)
    
    # Get mood values
    if selected_mood == "Custom":
        mood_values = {"danceability": 0.5, "energy": 0.5, "valence": 0.5, "tempo": 120.0, "acousticness": 0.5, "instrumentalness": 0.5, "liveness": 0.2, "loudness": -10.0, "key": 0}
    else:
        mood_values = MOOD_PRESETS[selected_mood]
        st.sidebar.info(f"🎭 {selected_mood} mood active!")
    
    # Sliders
    danceability = st.sidebar.slider('💃 Danceability', 0.0, 1.0, mood_values["danceability"])
    energy = st.sidebar.slider('⚡ Energy', 0.0, 1.0, mood_values["energy"])
    valence = st.sidebar.slider('😊 Valence', 0.0, 1.0, mood_values["valence"])
    tempo = st.sidebar.slider('🥁 Tempo', 60.0, 200.0, mood_values["tempo"])
    
    with st.sidebar.expander("🎛️ Advanced"):
        acousticness = st.slider('🎸 Acousticness', 0.0, 1.0, mood_values["acousticness"])
        instrumentalness = st.slider('🎹 Instrumentalness', 0.0, 1.0, mood_values["instrumentalness"])
        liveness = st.slider('🎤 Liveness', 0.0, 1.0, mood_values["liveness"])
        loudness = st.slider('🔊 Loudness', -60.0, 0.0, mood_values["loudness"])
        key = st.selectbox('🎼 Key', list(range(12)), index=int(mood_values["key"]))
    
    # Current preferences
    st.write("**🎯 Current Preferences:**")
    pref_col1, pref_col2, pref_col3, pref_col4 = st.columns(4)
    with pref_col1:
        st.metric("💃", f"{danceability:.2f}")
    with pref_col2:
        st.metric("⚡", f"{energy:.2f}")
    with pref_col3:
        st.metric("😊", f"{valence:.2f}")
    with pref_col4:
        st.metric("🥁", f"{tempo:.0f}")
    
    # GET RECOMMENDATIONS - The only button that CAN refresh
    if st.button('🎵 Get Recommendations', type="primary", use_container_width=True):
        exclude_list = list(st.session_state.user_actions['liked']) + list(st.session_state.user_actions['disliked']) if exclude_heard else []
        
        with st.spinner('🎵 Finding songs...'):
            if recommender_type == 'Feature Sliders':
                user_preferences = {
                    'danceability': danceability, 'energy': energy, 'valence': valence,
                    'acousticness': acousticness, 'instrumentalness': instrumentalness,
                    'liveness': liveness, 'tempo': tempo, 'loudness': loudness, 'key': key
                }
                
                recommendations = recommend_by_features(user_preferences, songs_data, k, exclude_list)
                
                if not recommendations.empty:
                    st.success(f'🎵 Found {len(recommendations)} songs!')
                    st.session_state.current_recs = recommendations
                    display_recommendations(recommendations, show_scores=True)
                else:
                    st.warning("No songs found.")
            
            elif recommender_type == 'Content-Based Filtering':
                if song_name and artist_name:
                    song_name_clean = song_name.lower().strip()
                    artist_name_clean = artist_name.lower().strip()
                    
                    if ((songs_data["name"] == song_name_clean) & (songs_data['artist'] == artist_name_clean)).any():
                        try:
                            recommendations = content_recommendation(song_name_clean, artist_name_clean, songs_data, transformed_data, k)
                            st.success(f'🎵 Found songs similar to **{song_name}** by **{artist_name}**')
                            st.session_state.current_recs = recommendations
                            display_recommendations(recommendations)
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.warning("Song not found in database")
                else:
                    st.warning("Please enter song and artist name")
    
    # Display current recommendations if they exist (after page refresh)
    if not st.session_state.current_recs.empty and 'recommendations' not in locals():
        st.write("### 🎵 Current Recommendations:")
        display_recommendations(st.session_state.current_recs, show_scores=True)

# PLAYLISTS PAGE
elif page == "📝 Playlists":
    st.header("📝 My Playlists")
    
    # Create playlist
    col1, col2 = st.columns([3, 1])
    with col1:
        new_playlist = st.text_input("🆕 Create playlist:")
    with col2:
        if st.button("➕ Create"):
            if new_playlist and new_playlist not in st.session_state.user_actions['playlists']:
                st.session_state.user_actions['playlists'][new_playlist] = []
                st.success(f"Created: {new_playlist}")
    
    # Display playlists
    if st.session_state.user_actions['playlists']:
        for name, songs in st.session_state.user_actions['playlists'].items():
            with st.expander(f"📋 {name} ({len(songs)} songs)"):
                if songs:
                    for i, song in enumerate(songs):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{song['name']}** by {song['artist']}")
                        with col2:
                            if st.button("🗑️", key=f"del_{name}_{i}"):
                                st.session_state.user_actions['playlists'][name].pop(i)
                                st.rerun()
                else:
                    st.write("Empty playlist")

# ANALYTICS PAGE
elif page == "📊 Analytics":
    st.header("📊 Analytics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("❤️ Liked", len(st.session_state.user_actions['liked']))
    with col2:
        st.metric("👎 Disliked", len(st.session_state.user_actions['disliked']))
    with col3:
        st.metric("🎵 Queued", len(st.session_state.user_actions['queued']))
    
    if st.session_state.user_actions['liked']:
        st.subheader("❤️ Liked Songs")
        for song in list(st.session_state.user_actions['liked']):
            parts = song.split('_')
            st.write(f"• **{parts[0]}** by {parts[1] if len(parts) > 1 else 'Unknown'}")

# SETTINGS PAGE
elif page == "⚙️ Settings":
    st.header("⚙️ Settings")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Clear Liked"):
            st.session_state.user_actions['liked'].clear()
            st.success("Cleared!")
    with col2:
        if st.button("🗑️ Clear Disliked"):
            st.session_state.user_actions['disliked'].clear()
            st.success("Cleared!")
    with col3:
        if st.button("🗑️ Clear Playlists"):
            st.session_state.user_actions['playlists'].clear()
            st.success("Cleared!")

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #1DB954; font-weight: bold;">🎵 **BULLETPROOF** Static Recommender - NO REFRESH GUARANTEED! 🎵</div>', unsafe_allow_html=True)
