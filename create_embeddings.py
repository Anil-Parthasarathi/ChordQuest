import pandas as pd
import numpy as np
import json
import re
import torch
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm
import gc

# --- CONFIGURATION ---
BATCH_SIZE = 64          # Increased slightly for GPU efficiency (Safe for 1080 Ti)
CHUNK_SIZE = 20000       # Rows to process before writing to disk
MODEL_NAME = 'all-MiniLM-L6-v2'

# Define Paths
BASE_PATH = Path("./backend/data/")
LASTFM_RAW_FILE = BASE_PATH / "raw/lastfm_dataset.csv"
MUSESCORE_RAW_FILE = BASE_PATH / "raw/score.jsonl"
PROCESSED_DIR = BASE_PATH / "processed"

LASTFM_OUTPUT = PROCESSED_DIR / "lastfm_embeddings.parquet"
MUSESCORE_OUTPUT = PROCESSED_DIR / "musescore_embeddings.parquet"

# --- 1. TEXT SANITIZATION ---
def sanitize_text(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    # Remove content inside brackets/parentheses
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    
    # Remove noise words that confuse semantic matching
    noise = [r'ft\.', r'feat\.', r'official', r'instrumental', r'acoustic', 
             r'remaster(ed)?', r'version', r'cover', r'solo', r'duet', 
             r'arrang(ement)?', r'sheet music']
    for pattern in noise:
        text = re.sub(r'\b' + pattern + r'\b', '', text)
        
    text = re.sub(r'[^\w\s]', ' ', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Collapse whitespace
    return text

# --- 2. EMBEDDER CLASS ---
class Embedder:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading Model on device: {self.device}")
        self.model = SentenceTransformer(MODEL_NAME, device=self.device)
        # 128 is sufficient for titles, saves VRAM/Compute compared to 512
        self.model.max_seq_length = 128 

    def encode_batch(self, text_list):
        embeddings = self.model.encode(
            text_list,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=True 
        )
        return embeddings

# --- 3. STREAMING PROCESSORS ---

def process_lastfm_streaming(input_file, output_file, embedder):
    """Reads CSV in chunks, embeds, and writes to Parquet."""
    print(f"\n--- Processing Last.fm: {input_file} ---")
    
    writer = None
    # Read raw CSV
    chunk_iterator = pd.read_csv(
        input_file, 
        sep=',', 
        names=['user', 'track', 'artist', 'playcount'],
        dtype={'playcount': 'object'}, 
        chunksize=CHUNK_SIZE, 
        engine='python', 
        on_bad_lines='skip'
    )

    for i, chunk in enumerate(tqdm(chunk_iterator, desc="Last.fm Chunks")):
        # 1. Prepare Text
        chunk['track'] = chunk['track'].astype(str).fillna('')
        chunk['artist'] = chunk['artist'].astype(str).fillna('')
        
        chunk['s_track'] = chunk['track'].apply(sanitize_text)
        chunk['s_artist'] = chunk['artist'].apply(sanitize_text)
        
        # Format: "ARTIST: Queen. SONG: Bohemian Rhapsody."
        texts = ("ARTIST: " + chunk['s_artist'] + ". " + "SONG: " + chunk['s_track'] + ".").tolist()
        
        # 2. Embed
        embeddings = embedder.encode_batch(texts)
        chunk['embedding'] = list(embeddings)
        
        # 3. Cleanup & Write
        chunk.drop(columns=['s_track', 's_artist'], inplace=True)
        table = pa.Table.from_pandas(chunk)
        
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)
        
        # 4. Memory Management
        del embeddings, texts, chunk, table
        if embedder.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect() # Help system RAM

    if writer: 
        writer.close()
    print(f"Saved to {output_file}")


def process_musescore_streaming_robust(input_file, output_file, embedder):
    print(f"\n--- Processing Musescore: {input_file} ---")
    
    writer = None
    chunk_buffer = []

    # Schema includes description/instruments for Frontend Display, 
    # even though we don't use them for embedding anymore.
    master_schema = pa.schema([
        ('id', pa.string()),
        ('authorUserId', pa.string()),
        ('title', pa.string()),
        ('description', pa.string()),
        ('instrumentsNames', pa.list_(pa.string())), 
        ('url', pa.string()),
        ('embedding', pa.list_(pa.float32())) 
    ])
    
    def flush_chunk(buffer):
        nonlocal writer
        if not buffer: return
        
        df = pd.DataFrame(buffer)
        
        # 1. Sanitize & Prepare Text
        # CRITICAL FIX: We only sanitize and embed the TITLE.
        # We ignore description/instruments to prevent vector drift.
        raw_titles = df['title'].astype(str).fillna('').tolist()
        clean_titles = [sanitize_text(t) for t in raw_titles]
        
        # Format: "SONG: Bohemian Rhapsody." 
        # (Matches the semantic tag from Last.fm)
        texts = [f"SONG: {t}." for t in clean_titles]
        
        # 2. Embed
        embeddings = embedder.encode_batch(texts)
        df['embedding'] = list(embeddings)
        
        # 3. Force Types for Output
        for col in ['title', 'description', 'id', 'url', 'authorUserId']:
            df[col] = df[col].astype(str)
            
        table = pa.Table.from_pandas(df, schema=master_schema)
        
        if writer is None:
            writer = pq.ParquetWriter(output_file, master_schema)
            
        writer.write_table(table)
        
        # 4. Memory Management
        del df, embeddings, texts, raw_titles, clean_titles, table
        if embedder.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # Main Loop
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading Musescore"):
            if not line.strip(): continue
            try:
                entry = json.loads(line)
                row = {
                    'id': str(entry.get('id', '')),
                    'authorUserId': str(entry.get('authorUserId', '')),
                    'title': str(entry.get('title', '')),
                    'description': str(entry.get('description', '')),
                    'instrumentsNames': entry.get('instrumentsNames', []), 
                    'url': str(entry.get('url', ''))
                }
                chunk_buffer.append(row)
                
                if len(chunk_buffer) >= CHUNK_SIZE:
                    flush_chunk(chunk_buffer)
                    chunk_buffer = [] 
                    
            except json.JSONDecodeError:
                continue
                
    # Flush remaining
    if chunk_buffer:
        flush_chunk(chunk_buffer)
        
    if writer:
        writer.close()
    print(f"Saved to {output_file}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize Embedder once
    main_embedder = Embedder()
    
    # 1. Run Last.fm (Context: Artist + Song)
    # if LASTFM_RAW_FILE.exists():
    #     process_lastfm_streaming(LASTFM_RAW_FILE, LASTFM_OUTPUT, main_embedder)
    # else:
    #     print(f"Skipping Last.fm (File not found: {LASTFM_RAW_FILE})")
        
    # 2. Run Musescore (Context: Song Title Only)
    if MUSESCORE_RAW_FILE.exists():
        process_musescore_streaming_robust(MUSESCORE_RAW_FILE, MUSESCORE_OUTPUT, main_embedder)
    else:
        print(f"Skipping MuseScore (File not found: {MUSESCORE_RAW_FILE})")
        
    print("\nPipeline Complete.")