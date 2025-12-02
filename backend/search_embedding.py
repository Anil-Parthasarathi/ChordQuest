import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import pickle
import re
from metadata_loader import id_to_metadata as shared_metadata

# Load the embedding model (all-MiniLM-L6-v2)
model = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# MuseScore Embeddings Index (for search)
# ==========================================
embeddings_path = Path(__file__).parent / 'musescore_embeddings.parquet'
cache_path = Path(__file__).parent / 'musescore_metadata' / 'faiss_index_cache.pkl'
df = None
index = None
id_to_sheet = {}

# ==========================================
# Last.fm Embeddings Index (for fallback recommendations)
# ==========================================
lastfm_embeddings_path = Path(__file__).parent / 'lastfm_unique_works.parquet'
lastfm_cache_path = Path(__file__).parent / 'musescore_metadata' / 'lastfm_faiss_cache.pkl'
lastfm_index = None
lastfm_idx_to_work = {}  # FAISS index -> work_id

def build_faiss_index():
    global df, index, id_to_sheet
    
    print(f"Loading embeddings from {embeddings_path}")
    df = pd.read_parquet(embeddings_path)
    
    # Extract embeddings as numpy array
    embeddings = np.array(df['embedding'].tolist()).astype('float32')
    
    # Build FAISS index for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  
    index.add(embeddings)
    
    # Create mapping from array index to sheet data
    for idx, row in df.iterrows():
        id_to_sheet[idx] = row.to_dict()
    
    print(f"FAISS index built with {index.ntotal} embeddings")
    
    # Save to cache
    print("Saving FAISS index to cache...")
    cache_data = {
        'index': faiss.serialize_index(index),
        'id_to_sheet': id_to_sheet
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("FAISS index cached successfully!")

def initialize_faiss_index():
    global df, index, id_to_sheet
    
    if cache_path.exists():
        print(f"Loading cached FAISS index from {cache_path}")
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            index = faiss.deserialize_index(cache_data['index'])
            id_to_sheet = cache_data['id_to_sheet']
        print(f"FAISS index loaded successfully with {index.ntotal} embeddings!")
    else:
        build_faiss_index()


# ==========================================
# Last.fm Index Functions (for fallback)
# ==========================================
_lastfm_initialized = False

def build_lastfm_index():
    """Build FAISS index from Last.fm unique works embeddings."""
    global lastfm_index, lastfm_idx_to_work
    
    print(f"Building Last.fm FAISS index from {lastfm_embeddings_path}...")
    
    if not lastfm_embeddings_path.exists():
        print(f"  ✗ Last.fm embeddings not found at {lastfm_embeddings_path}")
        return False
    
    # Read the parquet file in chunks to reduce memory
    lastfm_df = pd.read_parquet(lastfm_embeddings_path)
    print(f"  Loaded {len(lastfm_df)} Last.fm works")
    
    # Check for required columns
    if 'embedding' not in lastfm_df.columns or 'work_id' not in lastfm_df.columns:
        print(f"  ✗ Missing required columns. Found: {lastfm_df.columns.tolist()}")
        return False
    
    # Extract embeddings
    embeddings = np.array(lastfm_df['embedding'].tolist()).astype('float32')
    faiss.normalize_L2(embeddings)
    
    # Build index
    dimension = embeddings.shape[1]
    lastfm_index = faiss.IndexFlatIP(dimension)
    lastfm_index.add(embeddings)
    
    # Build mapping from FAISS index to work_id
    lastfm_idx_to_work = {i: int(wid) for i, wid in enumerate(lastfm_df['work_id'].values)}
    
    # Free memory
    del lastfm_df, embeddings
    
    print(f"  Last.fm FAISS index built with {lastfm_index.ntotal} works")
    
    # Cache the index
    print("  Saving Last.fm index to cache...")
    cache_data = {
        'index': faiss.serialize_index(lastfm_index),
        'idx_to_work': lastfm_idx_to_work
    }
    with open(lastfm_cache_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("  Last.fm index cached!")
    
    return True


def _ensure_lastfm_index():
    """Lazy load the Last.fm FAISS index only when needed."""
    global lastfm_index, lastfm_idx_to_work, _lastfm_initialized
    
    if _lastfm_initialized:
        return lastfm_index is not None
    
    _lastfm_initialized = True
    
    if lastfm_cache_path.exists():
        print(f"Loading cached Last.fm FAISS index from {lastfm_cache_path}")
        try:
            with open(lastfm_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                lastfm_index = faiss.deserialize_index(cache_data['index'])
                lastfm_idx_to_work = cache_data['idx_to_work']
            print(f"  Last.fm index loaded with {lastfm_index.ntotal} works!")
            return True
        except Exception as e:
            print(f"  Failed to load cache: {e}")
    
    return build_lastfm_index()


# Initialize MuseScore index on module load 
initialize_faiss_index()
# NOTE: Last.fm index is lazy loaded on first use to save memory

def clean_title(title):
    """
    Remove extra information from titles to improve matching.
    Example: "Symphony No. 5 (Arr. for Piano)" -> "Symphony No. 5"
    """
    if not title:
        return ""
    # Remove content in parentheses and brackets
    title = re.sub(r'[\(\[].*?[\)\]]', '', title)
    # Remove extra whitespace
    return title.strip()

def embedding_search(query):
    """Main search function for user queries."""
    print(f"Embedding search for: {query}")
    
    # Generate embedding for the query
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)
    
    # Search for top 10k most similar
    k = min(10000, index.ntotal)
    scores, indices = index.search(query_embedding, k)
    
    # Prepare results in the same format as BM25 search
    top_results = []
    
    for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
        sheet_data = id_to_sheet[idx]
        sheet_id = sheet_data.get('id')
        
        # Get full metadata from shared loader
        full_metadata = shared_metadata.get(sheet_id, {})
        
        # Convert instrumentsNames from string to list if needed
        instruments = sheet_data.get('instrumentsNames', [])
        if isinstance(instruments, str):
            instruments = eval(instruments) if instruments else []
        
        top_results.append({
            'id': sheet_id,
            'title': sheet_data.get('title', 'Untitled'),
            'artist': sheet_data.get('authorUserId', 'Unknown'),
            'authorUserId': sheet_data.get('authorUserId', ''),
            'difficulty': 'N/A',
            'key': 'N/A',
            'url': sheet_data.get('url', ''),
            'score': float(score),  # Cosine similarity score
            'description': sheet_data.get('description', ''),
            'instrumentsNames': full_metadata.get('instrumentsNames', instruments),
            'pagesCount': full_metadata.get('pagesCount', 0),
            'partsCount': full_metadata.get('partsCount', 0),
            'partsNames': full_metadata.get('partsNames', []),
            'instrumentsCount': full_metadata.get('instrumentsCount', 0),
            'duration': full_metadata.get('duration', 0),
            'timeCreated': full_metadata.get('timeCreated', ''),
            'timeUpdated': full_metadata.get('timeUpdated', '')
        })
    
    return top_results


def find_work_id_by_title(title, k=1, allowed_work_ids=None):
    """
    Find the most similar Last.fm work_id for a given song title.
    
    Args:
        title: Song title to search for
        k: Number of results to return (if no filter)
        allowed_work_ids: Set of work_ids that are valid/known in the model.
                          If provided, we prioritize finding a match in this set.
        
    Returns:
        List of (work_id, similarity_score) tuples, or empty list if not available
    """
    if not title:
        return []
    
    # Lazy load the Last.fm index
    if not _ensure_lastfm_index():
        print("  ✗ Last.fm index not available")  
        return []
    
    if lastfm_index is None:
        print("  ✗ Last.fm index is None after ensure")  
        return []
    
    clean_t = clean_title(title)
    if not clean_t:
        clean_t = title # Fallback to original if cleaning removed everything
    
    print(f"  → Last.fm search for: '{clean_t}'")  
        
    # Generate embedding for the title
    query_embedding = model.encode([clean_t], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Search deeper to find valid matches
    search_k = 2000 if allowed_work_ids else min(k, lastfm_index.ntotal)
    search_k = min(search_k, lastfm_index.ntotal)
    
    scores, indices = lastfm_index.search(query_embedding, search_k)
    
    results = []
    found_valid = False
    
    # Pass 1: Look for high-quality matches in the allowed set
    if allowed_work_ids:
        print(f"  Searching through {len(indices[0])} candidates for valid work IDs")  
        valid_count = 0
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx in lastfm_idx_to_work:
                wid = lastfm_idx_to_work[idx]
                if wid in allowed_work_ids:
                    results.append((wid, float(score)))
                    found_valid = True
                    valid_count += 1
                    if len(results) >= k:
                        print(f"  Found {valid_count} valid matches, returning top {k}")  
                        return results
        
        if valid_count > 0:
            print(f"  Found {valid_count} valid matches total")  
        else:
            print(f"  No valid matches found in allowed_work_ids")  

    # Pass 2: Fallback (if no valid model ID found, or no filter provided)
    if not found_valid and not allowed_work_ids:
        results = [] # Reset
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx in lastfm_idx_to_work:
                results.append((lastfm_idx_to_work[idx], float(score)))
                if len(results) >= k:
                    break
    
    return results


def find_similar_sheet_with_mapping(sheet_id, sheet_to_work_mapping, k=50, allowed_work_ids=None):
    """
    Find a work_id for a sheet that doesn't have a direct mapping.
    
    Strategy:
    1. First, try searching the Last.fm index directly by title (most reliable)
       WITH the allowed_work_ids constraint to ensure we pick a "popular" song.
    2. Fall back to finding similar MuseScore sheets with mappings
    
    Args:
        sheet_id: The MuseScore sheet ID to find a similar match for
        sheet_to_work_mapping: Dict mapping sheet_id -> work_id
        k: Number of similar items to search through
        allowed_work_ids: Set of work_ids valid for the Recommender model
        
    Returns:
        (work_id, similarity_score) if found, (None, 0) otherwise
    """
    # Get the sheet's title from metadata
    sheet_metadata = shared_metadata.get(str(sheet_id), {})
    if not sheet_metadata:
        sheet_metadata = shared_metadata.get(sheet_id, {})
    
    title = sheet_metadata.get('title', '')
    if not title:
        return None, 0
    
    # Strategy 1: Search directly in Last.fm index by title
    # We pass the allowed_work_ids to prioritize songs the model actually knows
    lastfm_results = find_work_id_by_title(title, k=5, allowed_work_ids=allowed_work_ids)
    
    if lastfm_results:
        # Try to find the best valid match
        for work_id, score in lastfm_results:
            # Use tiered thresholds based on whether we have constraints
            threshold = 0.25 if allowed_work_ids else 0.35  # Lowered from 0.3/0.4
            if score >= threshold:
                return work_id, score
    
    # Strategy 2: Find similar MuseScore sheets with mappings
    if index is not None:
        # Use uncleaned title for exact sheet lookup
        query_embedding = model.encode([title], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        search_k = min(k, index.ntotal)
        scores, indices = index.search(query_embedding, search_k)
        
        for idx, score in zip(indices[0], scores[0]):
            candidate_sheet = id_to_sheet.get(idx, {})
            candidate_id = str(candidate_sheet.get('id', ''))
            
            if candidate_id == str(sheet_id):
                continue
            
            if candidate_id in sheet_to_work_mapping:
                wid = sheet_to_work_mapping[candidate_id]
                # If we have a filter, ensure the mapped work is valid
                if allowed_work_ids and wid not in allowed_work_ids:
                    continue
                    
                return wid, float(score)
    
    return None, 0


def find_similar_sheets_with_mapping_batch(sheet_ids, sheet_to_work_mapping, k=50):
    """
    Find similar sheets with work mappings for multiple sheet IDs.
    
    Args:
        sheet_ids: List of MuseScore sheet IDs
        sheet_to_work_mapping: Dict mapping sheet_id -> work_id
        k: Number of similar sheets to search per query
        
    Returns:
        Dict mapping original sheet_id -> (work_id, similarity_score)
    """
    results = {}
    
    for sheet_id in sheet_ids:
        sid_str = str(sheet_id)
        # Skip if already has a direct mapping
        if sid_str in sheet_to_work_mapping:
            results[sid_str] = (sheet_to_work_mapping[sid_str], 1.0)
        else:
            work_id, score = find_similar_sheet_with_mapping(sheet_id, sheet_to_work_mapping, k)
            if work_id is not None:
                results[sid_str] = (work_id, score)
    
    return results