import pandas as pd
import dask.dataframe as dd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Configuration ---
# Set the working directory using a raw string for the path
file_path = "D:/Project"

track_ids_save_path = os.path.join(file_path, "track_ids.npy")
filtered_data_save_path = os.path.join(file_path, "collab_filtered_data.csv")
interaction_matrix_save_path = os.path.join(file_path, "interaction_matrix.npz")

# Set input paths
songs_data_path = os.path.join(file_path, "cleaned_data.csv")
user_listening_history_data_path = os.path.join(file_path, "User Listening History.csv")


def filter_songs_data(songs_data: pd.DataFrame, track_ids: list, save_df_path: str) -> pd.DataFrame:
    """
    Filter the songs data for the given track ids and save it.
    """
    print("Filtering main songs data...")
    # Filter data based on track_ids
    filtered_data = songs_data[songs_data["track_id"].isin(track_ids)].copy()

    # Reset index
    filtered_data.reset_index(drop=True, inplace=True)
    
    # Pre-process song and artist names for easier matching later
    filtered_data['name'] = filtered_data['name'].str.lower().str.strip()
    filtered_data['artist'] = filtered_data['artist'].str.lower().str.strip()

    # Save the data
    save_pandas_data_to_csv(filtered_data, save_df_path)
    return filtered_data

def save_pandas_data_to_csv(data: pd.DataFrame, file_path: str) -> None:
    """
    Save the data to a csv file.
    """
    print(f"Saving DataFrame to {file_path}...")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data.to_csv(file_path, index=False)

def save_sparse_matrix(matrix: csr_matrix, file_path: str) -> None:
    """
    Save the sparse matrix to a npz file.
    """
    print(f"Saving sparse matrix to {file_path}...")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    save_npz(file_path, matrix)

def create_interaction_matrix(history_data: dd.DataFrame, track_ids_save_path: str, save_matrix_path: str) -> tuple[csr_matrix, np.ndarray]:
    """
    Creates and saves a user-item interaction matrix from listening history.
    Returns the sparse matrix and the list of track IDs.
    """
    print("Creating interaction matrix...")
    # Make a copy of data
    df = history_data.copy()

    # Convert the playcount column to float
    df['playcount'] = df['playcount'].astype(np.float64)

    # Convert string columns to categorical. This is a lazy operation in Dask.
    df = df.categorize(columns=['user_id', 'track_id'])

    # Get the actual track_id strings. Accessing .cat.categories on a Dask Series
    # triggers the computation to find the unique categories and returns a pandas Index.
    track_ids = df['track_id'].cat.categories.values
    user_ids = df['user_id'].cat.categories.values
    
    np.save(track_ids_save_path, track_ids, allow_pickle=True)
    print(f"Saved {len(track_ids)} unique track IDs.")

    # Get the integer codes for users and tracks. These are also Dask series.
    user_codes = df['user_id'].cat.codes
    track_codes = df['track_id'].cat.codes

    # Group by the integer codes and sum playcounts
    interaction_df = df.groupby([track_codes, user_codes])['playcount'].sum().reset_index()
    interaction_df.columns = ['track_idx', 'user_idx', 'playcount']

    # Compute the result from Dask
    interaction_computed = interaction_df.compute()

    # Get the dimensions for the sparse matrix from the computed categories
    n_tracks = len(track_ids)
    n_users = len(user_ids)

    # Create the sparse matrix
    interaction_matrix = csr_matrix(
        (interaction_computed['playcount'], (interaction_computed['track_idx'], interaction_computed['user_idx'])),
        shape=(n_tracks, n_users)
    )

    # Save the sparse matrix
    save_sparse_matrix(interaction_matrix, save_matrix_path)

    return interaction_matrix, track_ids

def get_collaborative_recommendations(song_name: str, artist_name: str, k: int, songs_data: pd.DataFrame, interaction_matrix: csr_matrix, track_ids: np.ndarray) -> pd.DataFrame:
    """
    Generates song recommendations based on collaborative filtering.
    """
    print(f"\nGenerating {k} recommendations for '{song_name}' by '{artist_name}'...")
    # Lowercase and strip input for matching
    song_name = song_name.lower().strip()
    artist_name = artist_name.lower().strip()

    # Fetch the row from songs data
    song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
    
    if song_row.empty:
        print("Error: Song not found in the dataset.")
        return pd.DataFrame()

    # Track_id of input song
    input_track_id = song_row['track_id'].values.item()

    # Index value of track_id in our interaction matrix
    try:
        ind = np.where(track_ids == input_track_id)[0].item()
    except (ValueError, IndexError):
        print("Error: Song found in metadata but not in interaction matrix.")
        return pd.DataFrame()

    # Fetch the input vector
    input_vector = interaction_matrix[ind]

    # Get similarity scores
    similarity_scores = cosine_similarity(input_vector, interaction_matrix)

    # Index values of top recommendations (k+1 to include the input song itself)
    recommendation_indices = np.argsort(similarity_scores.ravel())[-k-1:][::-1]

    # Get track_ids of recommended songs
    recommendation_track_ids = track_ids[recommendation_indices]
    
    # Get top scores
    top_scores = np.sort(similarity_scores.ravel())[-k-1:][::-1]
    
    # Create a DataFrame with recommended track_ids and their scores
    scores_df = pd.DataFrame({"track_id": recommendation_track_ids, "score": top_scores})
    
    # Merge with song metadata to get song names and artists
    top_k_songs = (
        songs_data
        .merge(scores_df, on="track_id")
        .sort_values(by="score", ascending=False)
    )

    # Exclude the input song itself from the recommendations
    top_k_songs = top_k_songs[top_k_songs['track_id'] != input_track_id]

    return (
        top_k_songs
        .drop(columns=["track_id", "score"])
        .reset_index(drop=True)
        .head(k)
    )

def main():
    """
    Main function to run the data processing and recommendation pipeline.
    """
    try:
        # --- Part 1: Data Preprocessing ---
        # Load the user listening history
        user_data = dd.read_csv(user_listening_history_data_path)

        # Create and save the interaction matrix and track IDs
        interaction_matrix, track_ids = create_interaction_matrix(user_data, track_ids_save_path, interaction_matrix_save_path)

        # Load the main songs metadata
        songs_data = pd.read_csv("cleaned_data.csv")
        
        # Filter the songs metadata to only include songs in our interaction matrix
        filtered_songs_data = filter_songs_data(songs_data, track_ids.tolist(), filtered_data_save_path)
        print("\nPreprocessing complete. All required files have been created.")

        # --- Part 2: Generate Recommendations (Example) ---
        # Now, use the generated files to get recommendations
        recommendations = get_collaborative_recommendations(
            song_name="Love Story",
            artist_name="Taylor Swift",
            k=10,
            songs_data=filtered_songs_data,
            interaction_matrix=interaction_matrix,
            track_ids=track_ids
        )

        if not recommendations.empty:
            print("\n--- Recommendations ---")
            print(recommendations)

    except FileNotFoundError as e:
        print(f"\nError: Input file not found. Please check the path: {e.filename}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()