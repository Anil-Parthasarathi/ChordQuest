import json
from pathlib import Path
import pickle

# Paths
metadata_path = Path(__file__).parent / 'musescore_metadata' / 'musescore_song_metadata.jsonl'
cache_path = Path(__file__).parent / 'musescore_metadata' / 'metadata_cache.pkl'

# Global metadata dictionary
id_to_metadata = {}

def load_metadata_from_jsonl():
    """Load metadata from JSONL file"""
    metadata = {}
    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            sheet = json.loads(line)
            if "id" in sheet:
                metadata[sheet["id"]] = sheet
    print(f"Loaded metadata for {len(metadata)} sheets")
    return metadata

def initialize_metadata():
    """Initialize metadata with caching"""
    global id_to_metadata
    
    if cache_path.exists():
        print(f"Loading cached metadata from {cache_path}")
        with open(cache_path, 'rb') as f:
            id_to_metadata = pickle.load(f)
        print(f"Metadata loaded successfully for {len(id_to_metadata)} sheets!")
    else:
        print("Building metadata cache...")
        id_to_metadata = load_metadata_from_jsonl()
        
        # Save to cache
        print("Saving metadata to cache...")
        with open(cache_path, 'wb') as f:
            pickle.dump(id_to_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Metadata cached successfully!")
    
    return id_to_metadata

def get_metadata(sheet_id):
    """Get metadata for a specific sheet ID"""
    return id_to_metadata.get(sheet_id, {})

# Initialize on module load
initialize_metadata()
