import pandas as pd
import numpy as np
from pathlib import Path
from metadata_loader import id_to_metadata
import json

# Path to Last.fm dataset
LASTFM_DATASET_PATH = Path(__file__).parent / 'lastfm_dataset.csv'

recvae_model = None


class DummyRecVAEModel:
    """
    Dummy RecVAE model for testing.
    Replace this with the actual trained RecVAE model later.
    """
    
    def __init__(self):
        self.lastfm_songs = []
        self.load_lastfm_data()
    
    def load_lastfm_data(self):
        """Load Last.fm dataset to have song titles available"""
        try:
            df = pd.read_csv(LASTFM_DATASET_PATH, usecols=['work_id'], nrows=10000)
            self.lastfm_songs = df.tolist()
            print(f"Loaded {len(self.lastfm_songs)} Last.fm songs for dummy model")
        except Exception as e:
            print(f"Warning: Could not load Last.fm data: {e}")
            self.lastfm_songs = []
    
    def predict(self, song_title, k=10):
        """
        Dummy prediction function.
        
        Args:
            song_title: The song title to get recommendations for
            k: Number of recommendations to return
            
        Returns:
            List of recommended Last.fm song titles
        """
        # For now, just return random songs from Last.fm dataset
        # TODO: Replace with actual RecVAE model predictions
        if not self.lastfm_songs:
            return []
        
        # Return random songs (excluding exact matches)
        recommendations = []
        for song in self.lastfm_songs:
            if song.lower() != song_title.lower():
                recommendations.append(song)
                if len(recommendations) >= k:
                    break
        
        return recommendations


# Take musecore id then get work id to map to lastfm
def find_similar_lastfm_title(musescore_title):
    """
    Find the most similar Last.fm song title for a given MuseScore title.
    
    TODO: Implement actual similarity matching (e.g., using fuzzy string matching,
    embeddings, or exact match lookup).
    
    Args:
        musescore_title: The MuseScore song title
        
    Returns:
        The most similar Last.fm song title
    """
    # For now, just return the title as-is
    # TODO: Implement proper matching logic
    return musescore_title

def find_musescore_songs_from_lastfm(lastfm_titles, k=10):
    """
    Map Last.fm song titles back to MuseScore songs.
    
    TODO: Implement similarity matching to find MuseScore songs that match
    the Last.fm recommendations.
    
    Args:
        lastfm_titles: List of Last.fm song titles
        k: Maximum number of MuseScore songs to return
        
    Returns:
        List of MuseScore song dictionaries with metadata
    """
    musescore_results = []
    
    # Get all MuseScore songs
    for song_id, metadata in id_to_metadata.items():
        if len(musescore_results) >= k:
            break
        
        title = metadata.get('title', '')
        
        # Simple matching: check if any Last.fm title contains keywords from this title
        # TODO: Implement better matching (fuzzy matching, embeddings, etc.)
        for lastfm_title in lastfm_titles:
            if title.lower() in lastfm_title.lower() or lastfm_title.lower() in title.lower():
                musescore_results.append({
                    'id': song_id,
                    'title': metadata.get('title', 'Untitled'),
                    'artist': metadata.get('authorUserId', 'Unknown'),
                    'authorUserId': metadata.get('authorUserId', ''),
                    'difficulty': 'N/A',
                    'key': 'N/A',
                    'url': metadata.get('url', ''),
                    'score': 0.0,  # Placeholder score
                    'description': metadata.get('description', ''),
                    'instrumentsNames': metadata.get('instrumentsNames', []),
                    'pagesCount': metadata.get('pagesCount', 0),
                    'partsCount': metadata.get('partsCount', 0),
                    'partsNames': metadata.get('partsNames', []),
                    'instrumentsCount': metadata.get('instrumentsCount', 0),
                    'duration': metadata.get('duration', 0),
                    'timeCreated': metadata.get('timeCreated', ''),
                    'timeUpdated': metadata.get('timeUpdated', '')
                })
                break
    
    # If we didn't find enough matches, add some random songs
    if len(musescore_results) < k:
        for song_id, metadata in id_to_metadata.items():
            if len(musescore_results) >= k:
                break
            
            # Skip if already added
            if any(s['id'] == song_id for s in musescore_results):
                continue
            
            musescore_results.append({
                'id': song_id,
                'title': metadata.get('title', 'Untitled'),
                'artist': metadata.get('authorUserId', 'Unknown'),
                'authorUserId': metadata.get('authorUserId', ''),
                'difficulty': 'N/A',
                'key': 'N/A',
                'url': metadata.get('url', ''),
                'score': 0.0,
                'description': metadata.get('description', ''),
                'instrumentsNames': metadata.get('instrumentsNames', []),
                'pagesCount': metadata.get('pagesCount', 0),
                'partsCount': metadata.get('partsCount', 0),
                'partsNames': metadata.get('partsNames', []),
                'instrumentsCount': metadata.get('instrumentsCount', 0),
                'duration': metadata.get('duration', 0),
                'timeCreated': metadata.get('timeCreated', ''),
                'timeUpdated': metadata.get('timeUpdated', '')
            })
    
    return musescore_results


def get_recvae_recommendations(musescore_titles, k=10):
    """
    Get recommendations using RecVAE model.
    
    Workflow:
    1. Convert MuseScore titles to Last.fm titles (find most similar)
    2. Use RecVAE model to get Last.fm recommendations
    3. Convert Last.fm recommendations back to MuseScore songs
    
    Args:
        musescore_titles: List of MuseScore song titles (or single title)
        k: Number of recommendations to return
        
    Returns:
        List of recommended MuseScore songs
    """
    global recvae_model
    
    # Initialize dummy model if needed
    if recvae_model is None:
        print("Initializing dummy RecVAE model...")
        recvae_model = DummyRecVAEModel()
    
    # Ensure musescore_titles is a list
    if isinstance(musescore_titles, str):
        musescore_titles = [musescore_titles]
    
    # Collect all Last.fm recommendations
    all_lastfm_recommendations = []
    
    for musescore_title in musescore_titles:
        # Step 1: Convert MuseScore title to Last.fm title
        lastfm_title = find_similar_lastfm_title(musescore_title)
        
        # Step 2: Get RecVAE recommendations for this Last.fm song
        lastfm_recs = recvae_model.predict(lastfm_title, k=k*2)  # Get more to ensure variety
        all_lastfm_recommendations.extend(lastfm_recs)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_lastfm_recs = []
    for rec in all_lastfm_recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_lastfm_recs.append(rec)
    
    # Step 3: Convert Last.fm recommendations back to MuseScore songs
    musescore_recommendations = find_musescore_songs_from_lastfm(unique_lastfm_recs, k=k)
    
    return musescore_recommendations


def get_recommendations_by_song_ids(song_ids, k=10):
    """
    Get recommendations based on MuseScore song IDs using RecVAE.
    
    Args:
        song_ids: List of MuseScore song IDs
        k: Number of recommendations to return
        
    Returns:
        List of recommended MuseScore songs
    """
    if not song_ids:
        return []
    
    # Get titles for the song IDs
    musescore_titles = []
    for song_id in song_ids:
        metadata = id_to_metadata.get(song_id)
        if metadata:
            musescore_titles.append(metadata.get('title', ''))
    
    if not musescore_titles:
        return []
    
    # Get recommendations
    recommendations = get_recvae_recommendations(musescore_titles, k=k)
    
    # Filter out the input songs
    input_song_ids = set(song_ids)
    recommendations = [rec for rec in recommendations if rec['id'] not in input_song_ids]
    
    return recommendations[:k]


if __name__ == '__main__':
    # Test the recommendation system
    print("Testing RecVAE recommendation system...")
    
    # Test with a sample song ID (you can replace with actual ID)
    sample_song_ids = list(id_to_metadata.keys())[:1]
    
    if sample_song_ids:
        print(f"\nGetting recommendations for song ID: {sample_song_ids[0]}")
        recommendations = get_recommendations_by_song_ids(sample_song_ids, k=5)
        
        print(f"\nFound {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']}")
    else:
        print("No songs found in metadata")
