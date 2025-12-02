import polars as pl
import json
import shutil
import os
import glob
from pathlib import Path
import psutil
import re
import matplotlib.pyplot as plt
import gc
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import RecVAE
from recbole.trainer import RecVAETrainer
from recbole.utils import init_seed, init_logger

# --- CONFIGURATION ---
BASE_PATH = Path(__file__).parent.resolve()
INPUT_INTERACTIONS = BASE_PATH / "lastfm_embeddings.parquet"
INPUT_UNIQUE_MAP = BASE_PATH / "lastfm_unique_works.parquet"

DATASET_NAME = "lastfm_local"
RECBOLE_ROOT = Path("dataset") / DATASET_NAME
RECBOLE_INTER_FILE = RECBOLE_ROOT / f"{DATASET_NAME}.inter"

OUTPUT_DIR = BASE_PATH / "recvae_model_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def report_memory(stage=""):
    mem = psutil.virtual_memory()
    print(f"[{stage}] RAM: {mem.used/1024**3:.1f}GB Used | {mem.available/1024**3:.1f}GB Free")

def prepare_data():
    print("\n=== STEP 1: Preparing Data (Local) ===")
    report_memory("Start")
    
    if RECBOLE_ROOT.exists(): shutil.rmtree(RECBOLE_ROOT)
    RECBOLE_ROOT.mkdir(parents=True)

    print(f"--> Loading {INPUT_UNIQUE_MAP.name}...")
    q_map = pl.read_parquet(INPUT_UNIQUE_MAP).select(["artist", "track", "work_id"])
    
    print(f"--> Loading {INPUT_INTERACTIONS.name}...")
    q_inter = pl.read_parquet(INPUT_INTERACTIONS).select(["user", "artist", "track"])
    
    print("--> Linking User History to Work IDs...")
    df = (
        q_inter.join(q_map, on=["artist", "track"], how="inner")
        .select(["user", "work_id"])
    )
    
    print(f"--> Total Interactions: {len(df):,}")
    
    print("--> Remapping IDs...")
    
    # User Map
    unique_users = df["user"].unique().sort()
    user_map = dict(zip(unique_users.to_list(), range(1, len(unique_users) + 1)))
    
    # Item Map
    unique_items = df["work_id"].unique().sort()
    item_map = dict(zip(unique_items.to_list(), range(1, len(unique_items) + 1)))
    
    print(f"--> Vocabulary Size: {len(unique_items):,} Items")
    
    print(f"--> Saving intermediate file to {RECBOLE_INTER_FILE}...")
    df_final = df.with_columns([
        pl.col("user").replace(user_map).alias("user_id:token"),
        pl.col("work_id").replace(item_map).alias("item_id:token")
    ]).select(["user_id:token", "item_id:token"])
    
    df_final.write_csv(RECBOLE_INTER_FILE, separator="\t")
    
    print("--> Saving JSON Maps...")
    with open(OUTPUT_DIR / "user_map.json", "w") as f:
        json.dump(user_map, f)
        
    inv_item_map = {str(v): k for k, v in item_map.items()}
    with open(OUTPUT_DIR / "item_map_reverse.json", "w") as f:
        json.dump(inv_item_map, f)
        
    # Cleanup RAM
    del df, df_final, user_map, item_map, q_map, q_inter
    import gc; gc.collect()
    print("Data Preparation Complete.")

def trainer():
    print("\n=== STEP 2: Training RecVAE (Optimized for 8GB GPU) ===")
    
    gc.collect()
    torch.cuda.empty_cache()

    config_dict = {
        # --- DATA PATHS ---
        "data_path": "./dataset/",
        "dataset": DATASET_NAME,
        "field_separator": "\t",
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "load_col": {'inter': ['user_id', 'item_id']},
        
        "user_inter_num_interval": "[10,inf)", 
        "item_inter_num_interval": "[15,inf)", 
        
        "train_batch_size": 128,
        "eval_batch_size": 128,
        "fp16": True,
        
        "learning_rate": 0.001, 
        "hidden_dimension": 600,  
        "latent_dimension": 200,

        "epochs": 30,            
        "eval_step": 3,           
        "stopping_step": 5,       
        
        "dropout_prob": 0.5,
        "beta": 0.2,
        "gamma": 0.005,
        "learner": "adam",
        
        "eval_args": {
            "split": {'RS': [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": "full"
        },
        "metrics": ["NDCG", "Recall", "Hit"],
        "topk": [10, 20],
        "valid_metric": "NDCG@10",
        
        "use_gpu": True,
        "gpu_id": 0,
        "state": "INFO"
    }
    
    config = Config(model='RecVAE', config_dict=config_dict)
    
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    model = RecVAE(config, train_data.dataset).to(config['device'])
    
    trainer = RecVAETrainer(config, model)
    
    print("--> Starting Training Loop...")
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    
    print(f"--> Training Complete. Best Score: {best_valid_score}")
    
    test_result = trainer.evaluate(test_data)
    print("--> Test Results:", test_result)

def visualize_training():
    print("\n=== STEP 3: Generating Graphs ===")
    
    log_files = glob.glob('log/**/*.log', recursive=True)
    if not log_files: return

    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Parsing Log: {latest_log}")

    epochs, losses, ndcg_scores, recall_scores = [], [], [], []
    loss_pattern = re.compile(r"Training: \[epoch\s*:\s*(\d+).*train_loss:\s*([\d\.]+)")
    metrics_pattern = re.compile(r"Evaluating:.*NDCG@10:\s*([\d\.]+).*Recall@10:\s*([\d\.]+)")

    with open(latest_log, 'r') as f:
        for line in f:
            loss_match = loss_pattern.search(line)
            if loss_match:
                epochs.append(int(loss_match.group(1)))
                losses.append(float(loss_match.group(2)))
            
            metric_match = metrics_pattern.search(line)
            if metric_match:
                ndcg_scores.append(float(metric_match.group(1)))
                recall_scores.append(float(metric_match.group(2)))

    # Align lengths
    min_len = min(len(epochs), len(losses), len(ndcg_scores))
    epochs = epochs[:min_len]
    losses = losses[:min_len]
    ndcg_scores = ndcg_scores[:min_len]
    recall_scores = recall_scores[:min_len]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'RecVAE Training Results', fontsize=16)

    ax1.plot(epochs, losses, 'b-o', label='Loss')
    ax1.set_title('Training Loss (Lower is Better)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, ndcg_scores, 'g-s', label='NDCG@10')
    ax2.plot(epochs, recall_scores, 'r-^', label='Recall@10')
    ax2.set_title('Validation Metrics (Higher is Better)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plot_path = OUTPUT_DIR / "training_history.png"
    plt.savefig(plot_path)
    print(f"Graph saved to: {plot_path}")
    plt.close()

def save_best_model():
    
    list_of_files = glob.glob('saved/*.pth')
    if not list_of_files:
        print("Error: No model found in 'saved/'")
        return
        
    latest_file = max(list_of_files, key=os.path.getctime)
    dest_path = OUTPUT_DIR / "recvae_weights.pth"
    
    print(f"Moving {latest_file} -> {dest_path}")
    shutil.copy(latest_file, dest_path)
    
    print("\n✅ PIPELINE COMPLETE!")
    print(f"Model & Maps are located in: {OUTPUT_DIR}")

if __name__ == "__main__":
    if not INPUT_INTERACTIONS.exists():
        print(f"❌ Error: Could not find {INPUT_INTERACTIONS}")
    else:
        prepare_data()
        trainer()
        visualize_training()
        save_best_model()
