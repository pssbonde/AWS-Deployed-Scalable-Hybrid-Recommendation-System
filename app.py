import streamlit as st
import pandas as pd
import requests
import uuid

# --- FastAPI Backend URL ---
BACKEND_URL = "http://localhost:8001"

# --- User Session ---
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = str(uuid.uuid4())

# --- API Functions ---

def get_recommendations(song_name, artist_name, recommender_type, k):
    try:
        response = requests.post(
            f"{BACKEND_URL}/recommendations/",
            json={
                "song_name": song_name,
                "artist_name": artist_name,
                "recommender_type": recommender_type,
                "k": k,
            },
        )
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting recommendations: {e}")
        return pd.DataFrame()

def post_interaction(song_id, interaction_type):
    try:
        response = requests.post(
            f"{BACKEND_URL}/interactions/",
            json={
                "user_id": st.session_state.user_id,
                "song_id": song_id,
                "interaction_type": interaction_type,
            },
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Error posting interaction: {e}")

# --- UI ---

st.markdown("""
<style>
.main-header {
    font-size: 3rem; font-weight: bold; text-align: center;
    background: linear-gradient(90deg, #1DB954, #1ed760);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.2rem; border-radius: 15px;
    color: white; text-align: center;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    margin-bottom: 0.8rem;
}
.song-card {
    background: linear-gradient(135deg, #fff 60%, #e2ffe9 100%);
    padding: 1.25rem 1.2rem 1.2rem 1.2rem; border-radius: 16px;
    margin: 1.1rem 0;
    border-left: 5px solid #1DB954;
    box-shadow: 0 5px 18px rgba(29, 185, 84, 0.10);
}
.song-title { font-size: 1.15rem; font-weight: 700; color: #1DB954; margin-bottom: .18em;}
.song-artist { font-size: 1rem;  color: #444;  margin-bottom: .4em;}
.song-match { font-size: 0.93rem; color: #22bb77; font-weight: 600; }
.status-badges { margin: 0.1rem 0 0.5rem 0;}
.status-badge {
    padding: 0.25rem 0.85rem; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600; display:inline-block;
    margin-right: 0.5rem; margin-bottom:0.22em;
}
.badge-liked { background: #fff0f1; color: #ff6b6b;}
.badge-disliked { background: #e0f5ff; color: #3498db;}
.badge-queued { background: #f3f0ff; color: #6c5ce7; }
.now-playing {
    background: linear-gradient(90deg, #1DB954, #1ed760);
    color: white; padding: 0.45rem 1.2rem; border-radius: 18px;
    font-size: 0.99rem; font-weight: 500;
    margin-bottom: 0.65rem; display:inline-block;
}
.next-up {
    background: #eafaf2;
    color: #1DB954; padding: 0.36rem 1.05rem; border-radius: 17px;
    font-size: 0.92rem; font-weight: 600;
    margin-bottom: 0.52rem; display:inline-block;
}
.audio-preview { margin-top:0.7rem; }
.audio-available { color: #1DB954; font-weight: 500;}
.audio-unavailable { color: #9ca3af; font-style: italic;}
/* BUTTON: style ALL st.button globally for spacing and appearance */
.stButton > button {
    margin-right: 13px; margin-left: 2px;
    padding: 0.5rem 1.2rem !important;
    border-radius: 16px !important;
    background: linear-gradient(90deg, #1DB954, #1ed760) !important;
    color: white; font-weight: 600; font-size:1rem;
    border: none; box-shadow: 0 1.5px 7px rgba(29,185,84,.09);
    transition: all 0.18s;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #38e44e, #4cffde) !important;
    color: #212121;
}
</style>
""", unsafe_allow_html=True)

# ------ SESSION STATE ------
for key, default in {
    'liked_songs': set(),
    'disliked_songs': set(),
    'queued_songs': set(),
    'current_recommendations': pd.DataFrame(),
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -------- CALLBACKS --------
def like_callback(song_id):
    st.session_state.liked_songs.add(song_id)
    st.session_state.disliked_songs.discard(song_id)
    post_interaction(song_id, "like")

def dislike_callback(song_id):
    st.session_state.disliked_songs.add(song_id)
    st.session_state.liked_songs.discard(song_id)
    post_interaction(song_id, "dislike")

def queue_callback(song_id):
    st.session_state.queued_songs.add(song_id)
    post_interaction(song_id, "queue")

def display_static_recommendations(recommendations, show_scores=False):
    if recommendations.empty:
        st.warning("🎵 No recommendations found. Try adjusting your preferences!")
        return
    st.session_state.current_recommendations = recommendations
    for ind, rec in recommendations.iterrows():
        song_name = rec['name'].title()
        artist_name = rec['artist'].title()
        song_key = f"{rec['id']}" # Use song ID as key
        st.markdown('<div class="song-card">', unsafe_allow_html=True)
        if ind == 0:
            st.markdown('<div class="now-playing">🎵 Currently Playing</div>', unsafe_allow_html=True)
        elif ind == 1:
            st.markdown('<div class="next-up">🎶 Next Up</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="song-title">{song_name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="song-artist">by {artist_name}</div>', unsafe_allow_html=True)

        status_html = '<div class="status-badges">'
        if song_key in st.session_state.liked_songs:
            status_html += '<span class="status-badge badge-liked">💚 Liked</span>'
        if song_key in st.session_state.disliked_songs:
            status_html += '<span class="status-badge badge-disliked">💔 Disliked</span>'
        if song_key in st.session_state.queued_songs:
            status_html += '<span class="status-badge badge-queued">✅ Queued</span>'
        status_html += '</div>'
        st.markdown(status_html, unsafe_allow_html=True)

        cols = st.columns(3)
        if cols[0].button("❤️ Like", key=f"like_{ind}_{song_key}"):
            like_callback(song_key)
        if cols[1].button("👎 Pass", key=f"dislike_{ind}_{song_key}"):
            dislike_callback(song_key)
        if cols[2].button("➕ Queue", key=f"queue_{ind}_{song_key}"):
            queue_callback(song_key)

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

st.sidebar.markdown('<h1 style="color: #1DB954;">🎵 Navigation</h1>', unsafe_allow_html=True)
page = st.sidebar.selectbox("Choose Page", ["🏠 Home"])

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🎵 Hybrid Music Recommender</h1>', unsafe_allow_html=True)
    st.write('### 🎧 Discover your next favorite song')
    col1, col2 = st.columns(2)
    with col1:
        song_name = st.text_input('🎵 Enter a song name:')
    with col2:
        artist_name = st.text_input('🎤 Enter the artist name:')
    col1, col2 = st.columns(2)
    with col1:
        k = st.selectbox('📊 Number of recommendations:', [5, 10, 15, 20], index=1)
    with col2:
        recommender_type = st.selectbox('🤖 Algorithm:', ['collaborative'], index=0)

    if st.button('🎵 Get Recommendations', type="primary", use_container_width=True):
        if song_name and artist_name:
            with st.spinner('🎵 Finding your perfect songs...'):
                recommendations = get_recommendations(song_name, artist_name, recommender_type, k)
                if not recommendations.empty:
                    st.success(f'🎵 Found songs for you!')
                    display_static_recommendations(recommendations)
                else:
                    st.warning("Could not retrieve recommendations.")
        else:
            st.warning("Please enter a song name and artist.")

    elif not st.session_state.current_recommendations.empty:
        st.markdown("### 🎵 Your Last Recommendations")
        display_static_recommendations(st.session_state.current_recommendations)

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #1DB954; font-weight: bold;">'
    '🎵 Hybrid Music Recommender'
    '</div>',
    unsafe_allow_html=True
)
