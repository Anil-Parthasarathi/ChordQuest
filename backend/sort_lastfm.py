import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import faiss
import polars as pl
from scipy.sparse import csr_matrix, csgraph
from pathlib import Path
import gc
import json
import sys
import psutil
from tqdm import tqdm
import os

# --- CONFIGURATION ---
BASE_PATH = Path(__file__).parent.resolve()
INPUT_EMBEDDINGS = BASE_PATH / "lastfm_embeddings.parquet"
LASTFM_UNIQUE_FILE = BASE_PATH / "lastfm_unique_works.parquet"
MUSESCORE_INPUT = BASE_PATH / "musescore_embeddings.parquet"
MUSESCORE_TAGGED_FILE = BASE_PATH / "musescore_tagged.parquet"
API_OUTPUT_DIR = BASE_PATH / "api_lookups"

EMBEDDING_DIM = 384
CLUSTERING_THRESHOLD = 0.90 
MATCHING_THRESHOLD = 0.85   

TRAIN_SIZE = 200000
N_LIST = 4096     
N_PROBE = 5        
QUERY_BATCH_SIZE = 16384

def report_memory(stage):
    mem = psutil.virtual_memory()
    print(f"[{stage}] RAM Used: {mem.used / 1024**3:.2f}GB | Free: {mem.available / 1024**3:.2f}GB")

# ==========================================
# PHASE 1: DEDUPLICATION (Low RAM Mode)
# ==========================================
def step_1_deduplicate():
    # ... (Keeping this function for completeness, but we will skip it below) ...
    print("\n=== STEP 1: Deduplicating Last.fm Embeddings (Safe Mode) ===")
    report_memory("Start Step 1")
    
    if not INPUT_EMBEDDINGS.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_EMBEDDINGS}")

    seen_keys = set()
    writer = None
    
    pf = pq.ParquetFile(INPUT_EMBEDDINGS)
    
    for batch in tqdm(pf.iter_batches(batch_size=50000), desc="Deduplicating"):
        df = batch.to_pandas()
        keep_mask = []
        for artist, track in zip(df['artist'], df['track']):
            key = (artist, track)
            if key not in seen_keys:
                seen_keys.add(key)
                keep_mask.append(True)
            else:
                keep_mask.append(False)
        
        if any(keep_mask):
            df_unique = df[keep_mask]
            table = pa.Table.from_pandas(df_unique)
            if writer is None:
                writer = pq.ParquetWriter(LASTFM_UNIQUE_FILE, table.schema)
            writer.write_table(table)
            del df_unique, table
            
        del df, keep_mask
        
    if writer: writer.close()
    del seen_keys
    gc.collect()
    print("Step 1 Complete.")

# ==========================================
# PHASE 2: CLUSTERING (Fixed Write Type)
# ==========================================
def step_2_cluster_and_index():
    print("\n=== STEP 2: Clustering Unique Works (Safe Mode) ===")
    report_memory("Start Step 2")
    
    # 1. TRAIN INDEX
    print("--> Training GPU Index...")
    pf = pq.ParquetFile(LASTFM_UNIQUE_FILE)
    
    train_vectors = []
    current_count = 0
    total_rows = pf.metadata.num_rows
    actual_train_size = min(TRAIN_SIZE, total_rows)
    
    for batch in pf.iter_batches(batch_size=50000, columns=['embedding']):
        flat = batch['embedding'].values.to_numpy()
        rows = len(flat) // EMBEDDING_DIM
        vecs = flat.reshape((rows, EMBEDDING_DIM)).astype('float32')
        train_vectors.append(vecs)
        current_count += rows
        if current_count >= actual_train_size:
            break
            
    train_data = np.vstack(train_vectors)
    if len(train_data) > actual_train_size:
        train_data = train_data[:actual_train_size]
        
    faiss.normalize_L2(train_data)
    
    quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, N_LIST, faiss.METRIC_INNER_PRODUCT)
    index.train(train_data)
    
    del train_data, train_vectors
    gc.collect()
    index.nprobe = N_PROBE

    # 2. POPULATE INDEX
    print("--> Populating Index...")
    pf = pq.ParquetFile(LASTFM_UNIQUE_FILE)
    for batch in pf.iter_batches(batch_size=50000, columns=['embedding']):
        flat = batch['embedding'].values.to_numpy()
        rows = len(flat) // EMBEDDING_DIM
        chunk = flat.reshape((rows, EMBEDDING_DIM)).astype('float32')
        faiss.normalize_L2(chunk)
        index.add(chunk)
        del chunk
        
    print(f"Index Ready. Vectors: {index.ntotal}")

    # 3. SELF-SEARCH (GROUPING)
    print(f"--> Clustering (Threshold: {CLUSTERING_THRESHOLD})...")
    row_indices_list = []
    col_indices_list = []
    global_offset = 0
    
    pf = pq.ParquetFile(LASTFM_UNIQUE_FILE)
    total_batches = (pf.metadata.num_rows // QUERY_BATCH_SIZE) + 1
    
    with tqdm(total=total_batches, desc="Clustering") as pbar:
        for batch in pf.iter_batches(batch_size=QUERY_BATCH_SIZE, columns=['embedding']):
            flat = batch['embedding'].values.to_numpy()
            n_rows = len(flat) // EMBEDDING_DIM
            query = flat.reshape((n_rows, EMBEDDING_DIM)).astype('float32')
            faiss.normalize_L2(query)
            
            D, I = index.search(query, 50) 
            
            mask = D >= CLUSTERING_THRESHOLD
            local_rows, _ = np.nonzero(mask)
            valid_neighbors = I[mask]
            global_rows = local_rows + global_offset
            
            if len(global_rows) > 0:
                row_indices_list.append(global_rows.astype('int64'))
                col_indices_list.append(valid_neighbors.astype('int64'))
            
            global_offset += n_rows
            del query, D, I, mask
            pbar.update(1)
            
    # 4. BUILD GRAPH
    print("--> Generating work_ids...")
    all_rows = np.concatenate(row_indices_list)
    all_cols = np.concatenate(col_indices_list)
    del row_indices_list, col_indices_list
    gc.collect()
    
    graph = csr_matrix((np.ones(len(all_rows), dtype=bool), (all_rows, all_cols)), 
                       shape=(global_offset, global_offset))
    
    n_components, labels = csgraph.connected_components(graph, directed=False, connection='weak')
    print(f"Identified {n_components} Distinct Musical Works.")
    
    del graph, all_rows, all_cols
    gc.collect()

    # 5. SAVE IDs (FIXED TYPE ERROR HERE)
    print("--> Updating Unique File with IDs (Streaming Mode)...")
    
    temp_file = LASTFM_UNIQUE_FILE.with_suffix(".tmp.parquet")
    pf = pq.ParquetFile(LASTFM_UNIQUE_FILE)
    writer = None
    
    current_idx = 0
    
    for batch in pf.iter_batches(batch_size=50000):
        batch_len = len(batch)
        label_chunk = labels[current_idx : current_idx + batch_len]
        current_idx += batch_len
        
        work_id_array = pa.array(label_chunk.astype('int32'))
        new_batch = batch.append_column('work_id', work_id_array)
        
        # --- FIX: Convert RecordBatch to Table ---
        table_chunk = pa.Table.from_batches([new_batch])
        
        if writer is None:
            writer = pq.ParquetWriter(temp_file, table_chunk.schema)
        
        writer.write_table(table_chunk)
    
    if writer: writer.close()
    
    if LASTFM_UNIQUE_FILE.exists():
        os.remove(LASTFM_UNIQUE_FILE)
    temp_file.rename(LASTFM_UNIQUE_FILE)
    
    print("Saved! IDs updated successfully.")
    
    return labels, index

# ==========================================
# PHASE 3: MATCHING (Bridge Building)
# ==========================================
def step_3_match_musescore(work_ids, index):
    print("\n=== STEP 3: Building Bridge (MuseScore -> Last.fm) ===")
    report_memory("Start Step 3")
    
    if not MUSESCORE_INPUT.exists():
        print("MuseScore embeddings not found. Skipping Phase 3.")
        return

    reader = pq.ParquetFile(MUSESCORE_INPUT)
    writer = None
    
    output_schema = pa.schema([
        ('id', pa.string()),
        ('work_id', pa.int32()),
        ('match_score', pa.float32())
    ])
    
    total_batches = (reader.metadata.num_rows // QUERY_BATCH_SIZE) + 1
    
    with tqdm(total=total_batches, desc="Matching") as pbar:
        for batch in reader.iter_batches(batch_size=QUERY_BATCH_SIZE):
            df = batch.to_pandas()
            
            if 'embedding' not in df.columns: 
                pbar.update(1); continue
            
            valid_mask = df['embedding'].apply(lambda x: x is not None and len(x) == EMBEDDING_DIM)
            valid_rows = df[valid_mask]
            
            if valid_rows.empty:
                pbar.update(1); continue
            
            query = np.stack(valid_rows['embedding'].values).astype('float32')
            faiss.normalize_L2(query)
            
            distances, indices = index.search(query, k=1)
            
            best_scores = distances.flatten()
            best_indices = indices.flatten()
            
            found_ids = np.full(len(best_indices), -1, dtype='int32')
            valid_idx_mask = best_indices != -1
            found_ids[valid_idx_mask] = work_ids[best_indices[valid_idx_mask]]
            
            final_ids = np.where(best_scores >= MATCHING_THRESHOLD, found_ids, -1)
            
            output_df = valid_rows[['id']].copy()
            output_df['work_id'] = final_ids
            output_df['match_score'] = best_scores
            
            table = pa.Table.from_pandas(output_df, schema=output_schema)
            if writer is None:
                writer = pq.ParquetWriter(MUSESCORE_TAGGED_FILE, output_schema)
            writer.write_table(table)
            
            pbar.update(1)
            
    if writer: writer.close()
    print(f"Bridge File Created: {MUSESCORE_TAGGED_FILE}")

# ==========================================
# PHASE 4: API ARTIFACTS
# ==========================================
def step_4_api_lookups():
    print("\n=== STEP 4: Generating API Lookups ===")
    API_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not MUSESCORE_TAGGED_FILE.exists():
        return

    df = pl.read_parquet(MUSESCORE_TAGGED_FILE).filter(pl.col("work_id") != -1)
    
    if df.height == 0:
        print("Warning: No matches found. API maps will be empty.")
        return

    reverse_map = dict(zip(df["id"].to_list(), df["work_id"].to_list()))
    with open(API_OUTPUT_DIR / "sheet_to_work.json", "w") as f:
        json.dump(reverse_map, f)
        
    forward_df = (
        df.sort("match_score", descending=True)
        .group_by("work_id")
        .agg(pl.col("id").head(50))
    )
    forward_map = {str(r[0]): r[1] for r in forward_df.iter_rows()}
    with open(API_OUTPUT_DIR / "work_to_sheets.json", "w") as f:
        json.dump(forward_map, f)
        
    print(f"Maps saved to {API_OUTPUT_DIR}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        # 1. SKIP STEP 1 (It is done and safe)
        #step_1_deduplicate()

        # 2. Cluster & Create IDs (Returns Index to reuse)
        work_ids_array, trained_index = step_2_cluster_and_index()
        
        # 3. Match MuseScore
        step_3_match_musescore(work_ids_array, trained_index)
        
        # 4. Cleanup GPU
        del trained_index
        gc.collect()
        
        # 5. Generate JSONs
        step_4_api_lookups()
        
        print("\nSUCCESS! Data Pipeline Complete.")
        
    except Exception as e:
        print(f"\nPipeline Failed: {e}")
        import traceback
        traceback.print_exc()