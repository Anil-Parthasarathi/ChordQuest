import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import pickle
from metadata_loader import id_to_metadata as shared_metadata

# Load the embedding model (all-MiniLM-L6-v2)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the embeddings data
embeddings_path = Path(__file__).parent / 'musescore_embeddings.parquet'
cache_path = Path(__file__).parent / 'musescore_metadata' / 'faiss_index_cache.pkl'
df = None
index = None
id_to_sheet = {}

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

# Initialize on module load
initialize_faiss_index()

def embedding_search(query):

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
        
        # Use full metadata if available, otherwise fall back to embedding data
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