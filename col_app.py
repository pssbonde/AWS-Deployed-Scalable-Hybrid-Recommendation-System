import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# --- Page & Path Configuration ---
st.set_page_config(
    page_title="Collaborative Music Recommender",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use Pathlib for robust path handling, assuming files are in the same directory
INTERACTION_MATRIX_PATH = Path("interaction_matrix.npz")
TRACK_IDS_PATH = Path("track_ids.npy")
SONGS_DATA_PATH = Path("cleaned_data.csv")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    /* Main Title Style */
    .big-title {
        font-size: 50px;
        font-weight: 800;
        color: #4682B4; /* SteelBlue */
        text-align: center;
        margin-bottom: 10px;
    }
    /* Subtitle Style */
    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #333;
        margin-bottom: 30px;
    }
    /* Recommendation Card Style */
    .song-card {
        background-color: #FFFFFF;
        padding: 20px;
        margin: 15px 0;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #E0E0E0;
        transition: transform 0.2s;
    }
    .song-card:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
    }
    /* Center the footer */
    .footer {
        text-align: center;
        color: gray;
        font-size: 14px;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading (with Caching) ---

@st.cache_data
def load_songs_metadata(file_path: Path) -> tuple[pd.DataFrame, list]:
    """Loads song metadata and extracts a unique, sorted list of artists."""
    if not file_path.exists():
        st.error(f"FATAL: Metadata file not found at {file_path}")
        st.stop()
    df = pd.read_csv(file_path)
    artists = sorted(df['artist'].unique())
    return df, artists

@st.cache_resource
def load_recommendation_data(matrix_path: Path, track_ids_path: Path) -> tuple[csr_matrix, np.ndarray]:
    """Loads the pre-computed interaction matrix and track IDs."""
    if not matrix_path.exists() or not track_ids_path.exists():
        st.error(f"FATAL: Recommendation data files not found. Please run the preprocessing script first.")
        st.stop()
    interaction_matrix = load_npz(matrix_path)
    track_ids = np.load(track_ids_path, allow_pickle=True)
    return interaction_matrix, track_ids

# --- Core Recommendation Logic ---

def recommend_songs(song_name: str, artist_name: str, songs_data: pd.DataFrame,
                    interaction_matrix: csr_matrix, track_ids: np.ndarray,
                    k: int = 10) -> pd.DataFrame:
    """Recommends songs using collaborative filtering."""
    song_row = songs_data[
        (songs_data["name"].str.lower() == song_name.lower()) &
        (songs_data["artist"].str.lower() == artist_name.lower())
    ]
    if song_row.empty:
        return pd.DataFrame()

    input_track_id = song_row['track_id'].values[0]
    try:
        input_track_idx = np.where(track_ids == input_track_id)[0][0]
    except IndexError:
        return pd.DataFrame() # Song not in interaction matrix

    input_vector = interaction_matrix[input_track_idx]
    similarity_scores = cosine_similarity(input_vector, interaction_matrix).ravel()

    # Get top-k recommendations, excluding the song itself
    top_indices = np.argsort(similarity_scores)[-(k+1):][::-1]
    top_indices = top_indices[top_indices != input_track_idx][:k]

    rec_track_ids = track_ids[top_indices]
    rec_scores = similarity_scores[top_indices]

    scores_df = pd.DataFrame({"track_id": rec_track_ids, "similarity": rec_scores})
    top_songs = songs_data.merge(scores_df, on="track_id").sort_values(by="similarity", ascending=False)
    
    # Ensure all required columns are present for display
    if 'spotify_preview_url' not in top_songs.columns:
        top_songs['spotify_preview_url'] = None
        
    return top_songs

# Load the data once
songs_df, artist_list = load_songs_metadata(SONGS_DATA_PATH)
interaction_matrix, track_ids = load_recommendation_data(INTERACTION_MATRIX_PATH, TRACK_IDS_PATH)

# --- Sidebar for User Input ---
with st.sidebar:
    st.header("👥 Find Your Next Vibe")
    st.markdown("Select an artist and song to get recommendations based on what similar listeners enjoy.")

    selected_artist = st.selectbox(
        label="1. Choose an artist",
        options=[""] + artist_list,
        format_func=lambda x: "Select an artist" if x == "" else x,
    )

    if selected_artist:
        available_songs = sorted(songs_df[songs_df['artist'] == selected_artist]['name'].unique())
        selected_song = st.selectbox(
            label="2. Choose a song",
            options=[""] + available_songs,
            format_func=lambda x: "Select a song" if x == "" else x,
        )
    else:
        selected_song = st.selectbox(label="2. Choose a song", options=[""], disabled=True)

    k = st.selectbox("3. Number of Recommendations", [5, 10, 15, 20], index=1)
    submitted = st.button("🎶 Get Recommendations", type="primary", use_container_width=True)

# --- Main Panel for Header and Results ---
st.markdown('<div class="big-title">🎧 Collaborative Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover music enjoyed by listeners with similar tastes!</div>', unsafe_allow_html=True)
st.divider()

if submitted:
    if selected_artist and selected_song:
        with st.spinner(f"Finding songs similar to '{selected_song}'..."):
            recommendations = recommend_songs(selected_song, selected_artist, songs_df, interaction_matrix, track_ids, k)

        if not recommendations.empty:
            st.success(f"Here are {k} recommendations for: 🎵 **{selected_song.title()}**")
            
            for _, row in recommendations.iterrows():
                with st.container():
                    st.markdown('<div class="song-card">', unsafe_allow_html=True)
                    st.markdown(f"### {row['name'].title()}")
                    st.markdown(f"**👤 Artist:** {row['artist'].title()}")
                    st.metric(label="Similarity Score", value=f"{row['similarity']:.2%}")
                    
                    if pd.notna(row['spotify_preview_url']):
                        st.audio(row['spotify_preview_url'], format='audio/mp3')
                    else:
                        st.warning("No audio preview available for this song.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(f"❌ Sorry, we couldn't find **{selected_song.title()}** by **{selected_artist}** in our listening data. Try another song.")
    else:
        st.error("Please select both an artist and a song from the sidebar.")
else:
    st.info("Select an artist and song from the sidebar to get started!")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='footer'> © 2025 Collaborative AI</div>", unsafe_allow_html=True)