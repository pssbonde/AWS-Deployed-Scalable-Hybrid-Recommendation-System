import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import os

# --- Import your custom recommendation functions ---
from hybrid_recommendations import HybridRecommenderSystem
from collaborative_filtering import get_collaborative_recommendations
from content_based_filtering import get_content_based_recommendations



@st.cache_data
def load_songs_data(path):
    """Loads and pre-processes the songs metadata."""
    df = pd.read_csv(path)
    df['name'] = df['name'].str.lower().str.strip()
    df['artist'] = df['artist'].str.lower().str.strip()
    return df

@st.cache_data
def load_sparse_matrix(path):
    """Loads a sparse matrix from a .npz file."""
    return load_npz(path)

@st.cache_data
def load_numpy_array(path):
    """Loads a numpy array from a .npy file."""
    return np.load(path, allow_pickle=True)

# --- Main App Logic ---
try:
    # --- 1. Define File Paths ---
    cleaned_data_path = "cleaned_data.csv"
    filtered_data_path = "collab_filtered_data.csv"
    transformed_content_path = "transformed_data.npz" # For content-based
    transformed_hybrid_path = "transformed_hybrid_data.npz"
    interaction_matrix_path = "interaction_matrix.npz"
    track_ids_path = "track_ids.npy"

    # --- 2. Load All Necessary Data ---
    songs_data = load_songs_data(cleaned_data_path)
    filtered_songs_data = load_songs_data(filtered_data_path)
    transformed_content_data = load_sparse_matrix(transformed_content_path)
    transformed_hybrid_data = load_sparse_matrix(transformed_hybrid_path)
    interaction_matrix = load_sparse_matrix(interaction_matrix_path)
    track_ids = load_numpy_array(track_ids_path)

    # --- 3. Streamlit User Interface ---
    st.title('Welcome to the Hybrid Music Recommender!')
    st.write('### Enter the name of a song and the recommender will suggest similar songs')
    st.markdown("---")

    # User Inputs
    song_name_input = st.text_input('Enter a song name:', key="song_name")
    artist_name_input = st.text_input('Enter the artist name:', key="artist_name")

    col1, col2 = st.columns(2)
    with col1:
        k = st.selectbox('How many recommendations?', [5, 10, 15, 20], index=1, key="k")
    with col2:
        filtering_type = st.selectbox(
            'Select the type of filtering:',
            ['Hybrid Recommender System', 'Collaborative Filtering', 'Content-Based Filtering'],
            index=0,
            key="filtering_type"
        )

    # --- 4. Recommendation Logic ---
    if st.button('Get Recommendations', key="get_recs"):
        if not song_name_input or not artist_name_input:
            st.warning("Please enter both a song name and an artist name.")
        else:
            song_name = song_name_input.lower().strip()
            artist_name = artist_name_input.lower().strip()
            
            recommendations = pd.DataFrame()
            
            # Use the appropriate dataset for checking if the song exists
            data_to_check = songs_data if filtering_type == 'Content-Based Filtering' else filtered_songs_data
            
            input_song_query = (data_to_check["name"] == song_name) & (data_to_check["artist"] == artist_name)
            
            if not input_song_query.any():
                st.error(f"Sorry, '{song_name_input}' by '{artist_name_input}' was not found. Please check the spelling or try another song.")
            else:
                st.success(f"Generating {filtering_type} recommendations for '{song_name_input}' by '{artist_name_input}'...")

                if filtering_type == 'Hybrid Recommender System':
                    recommender = HybridRecommenderSystem(
                        song_name=song_name,
                        artist_name=artist_name,
                        number_of_recommendations=k,
                        weight_content_based=0.5,
                        weight_collaborative=0.5,
                        songs_data=filtered_songs_data,
                        transformed_matrix=transformed_hybrid_data,
                        interaction_matrix=interaction_matrix,
                        track_ids=track_ids
                    )
                    recommendations = recommender.give_recommendations()

                elif filtering_type == 'Collaborative Filtering':
                    recommendations = get_collaborative_recommendations(
                        song_name=song_name,
                        artist_name=artist_name,
                        k=k,
                        songs_data=filtered_songs_data,
                        interaction_matrix=interaction_matrix,
                        track_ids=track_ids
                    )
                
                elif filtering_type == 'Content-Based Filtering':
                    recommendations = get_content_based_recommendations(
                        song_name=song_name,
                        songs_data=songs_data,
                        transformed_data=transformed_content_data,
                        k=k
                    )
                
                # --- 5. Display Recommendations ---
                if not recommendations.empty:
                    # Display the user's chosen song first
                    input_song_details = data_to_check[input_song_query].iloc[0]
                    st.markdown("## You Selected:")
                    st.markdown(f"#### **{input_song_details['name'].title()}** by **{input_song_details['artist'].title()}**")
                    if 'spotify_preview_url' in input_song_details and pd.notna(input_song_details['spotify_preview_url']):
                        st.audio(input_song_details['spotify_preview_url'])
                    else:
                        st.write("No audio preview available for this song.")
                    st.write('---')
                    
                    # Display the list of recommendations
                    st.markdown("### Here are your recommendations:")
                    for index, rec in recommendations.iterrows():
                        rec_song = rec['name'].title()
                        rec_artist = rec['artist'].title()
                        st.markdown(f"#### {index + 1}. **{rec_song}** by **{rec_artist}**")
                        if 'spotify_preview_url' in rec and pd.notna(rec['spotify_preview_url']):
                            st.audio(rec['spotify_preview_url'])
                        else:
                            st.write("No audio preview available.")
                        st.write("---")
                else:
                    st.warning("Could not generate recommendations for this song. It might not have enough data.")

# --- 6. Error Handling for File Loading ---
except FileNotFoundError as e:
    st.error(f"Fatal Error: A required data file was not found.")
    st.error(f"Please make sure '{e.filename}' is in the 'data' directory relative to the script.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
