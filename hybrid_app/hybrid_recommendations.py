import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommenderSystem:

    def __init__(self, song_name: str,
                 artist_name: str,
                 number_of_recommendations: int,
                 weight_content_based:float,
                 weight_collaborative:float,
                 songs_data, transformed_matrix,
                 interaction_matrix, track_ids):

        self.number_of_recommendations = number_of_recommendations
        self.song_name = song_name.lower()
        self.artist_name = artist_name.lower()
        self.weight_content_based = weight_content_based
        self.weight_collaborative = weight_collaborative
        self.songs_data = songs_data
        self.transformed_matrix = transformed_matrix
        self.interaction_matrix = interaction_matrix
        self.track_ids = track_ids


    def calculate_content_based_similarities(self,song_name, artist_name, songs_data,transformed_matrix):

        # filter out the song from data
        song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
        
        # Check if the song was found
        if song_row.empty:
            raise ValueError("Song not found in the dataset.")
            
        #get the index of song
        song_index = song_row.index[0]
        
        # generate the input vector
        input_vector = transformed_matrix[song_index].reshape(1,-1)
        
        # calculate similarity scores
        content_similarity_scores = cosine_similarity(input_vector, transformed_matrix)

        return content_similarity_scores


    def calculate_collaborative_filtering_similarities(self, song_name, artist_name, track_ids, songs_data, interaction_matrix):

        #fetch the row from songs data
        song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
        
        # Check if the song was found
        if song_row.empty:
            raise ValueError("Song not found in the dataset.")

        # track_id of input song
        input_track_id = song_row['track_id'].values.item()

        # index value of track_id
        ind = np.where(track_ids == input_track_id)[0].item()

        #fetch the input vector
        input_array = interaction_matrix[ind]

        # get similarity scores
        collaborative_similarity_scores = cosine_similarity(input_array, interaction_matrix)

        return collaborative_similarity_scores
    

    def normalize_similarities(self, similarity_scores):

        minimum = np.min(similarity_scores)
        maximum = np.max(similarity_scores)
        # Avoid division by zero if all scores are the same
        if maximum == minimum:
            return np.zeros(similarity_scores.shape)
        normalized_scores = (similarity_scores - minimum) / (maximum - minimum)
        return normalized_scores
    

    def weighted_combination(self, content_based_scores, collaborative_filtering_scores):
        weighted_scores = (self.weight_content_based * content_based_scores) + (self.weight_collaborative * collaborative_filtering_scores)
        return weighted_scores
    

    def give_recommendations (self):
        """
        Generates song recommendations by combining content-based and collaborative filtering scores.
        """
        try:
            # --- 1. Calculate and Normalize Content-Based Scores ---
            content_based_similarities = self.calculate_content_based_similarities(
                song_name=self.song_name, artist_name=self.artist_name,
                songs_data=self.songs_data, transformed_matrix=self.transformed_matrix
            )
            normalized_content_scores = self.normalize_similarities(content_based_similarities)
            content_df = pd.DataFrame({
                'track_id': self.songs_data['track_id'],
                'content_score': normalized_content_scores.flatten()
            })

            # --- 2. Calculate and Normalize Collaborative Filtering Scores ---
            collaborative_based_similarities = self.calculate_collaborative_filtering_similarities(
                song_name=self.song_name, artist_name=self.artist_name,
                track_ids=self.track_ids, songs_data=self.songs_data,
                interaction_matrix=self.interaction_matrix
            )
            normalized_collab_scores = self.normalize_similarities(collaborative_based_similarities)
        
            collab_df = pd.DataFrame({
                'track_id': self.track_ids,
                'collab_score': normalized_collab_scores.flatten()
            })

            # --- 3. Merge scores and calculate weighted score ---
    
            merged_scores = pd.merge(content_df, collab_df, on='track_id', how='inner')
            merged_scores['weighted_score'] = (self.weight_content_based * merged_scores['content_score']) + \
                                              (self.weight_collaborative * merged_scores['collab_score'])

            # --- 4. Generate Final Recommendations ---
            # Merge with song details to get names, artists, etc.
            final_recommendations = pd.merge(self.songs_data, merged_scores, on='track_id')

            # Get the track_id of the input song to exclude it from the final list.
            input_track_id = self.songs_data.loc[
                (self.songs_data["name"] == self.song_name) & 
                (self.songs_data["artist"] == self.artist_name)
            ]['track_id'].iloc[0]

            # Sort by score, remove the input song, and get the top N recommendations.
            top_k_songs = (
                final_recommendations[final_recommendations['track_id'] != input_track_id]
                .sort_values(by='weighted_score', ascending=False)
                .head(self.number_of_recommendations)
                .drop(columns=['track_id', 'content_score', 'collab_score', 'weighted_score'])
                .reset_index(drop=True)
            )

            return top_k_songs

        except (ValueError, IndexError) as e:
            print(f"Error generating recommendations: {e}")
            print(f"Could not find song '{self.song_name}' by '{self.artist_name}'. Please check the spelling.")
            return pd.DataFrame()

    

if __name__ == "__main__":
    try:
        # load the transformed data
        transformed_data = load_npz("transformed_hybrid_data.npz")

        # load the interaction matrix
        interaction_matrix = load_npz("interaction_matrix.npz")

        # load the track ids
        track_ids = np.load("track_ids.npy", allow_pickle=True)

        # load the songs data
        songs_data = pd.read_csv("collab_filtered_data.csv", usecols=["track_id", "name", "artist", "spotify_preview_url"])
        
        # --- ADDED FOR ROBUSTNESS ---
        # Pre-process song and artist names to lower case for consistent matching
        songs_data['name'] = songs_data['name'].str.lower().str.strip()
        songs_data['artist'] = songs_data['artist'].str.lower().str.strip()


        # create an instance of HybridRecommenderSystem
        hybrid_recommender = HybridRecommenderSystem(
            song_name="Love Story",
            artist_name="Taylor Swift",
            number_of_recommendations=10,
            weight_content_based=0.3,
            weight_collaborative=0.7,
            songs_data=songs_data,
            transformed_matrix=transformed_data,
            interaction_matrix=interaction_matrix,
            track_ids=track_ids
        )

        # get recommendations
        recommendations = hybrid_recommender.give_recommendations()
        
        if not recommendations.empty:
            print("--- Recommendations ---")
            print(recommendations)

    except FileNotFoundError as e:
        print(f"Error: A required data file was not found.")
        print(f"Please make sure '{e.filename}' is in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
