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
from recbole.model.general_recommender import RecVAE, BPR
from recbole.trainer import RecVAETrainer, Trainer
from recbole.utils import init_seed, init_logger

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
    unique_users = df["user"].unique().sort()
    user_map = dict(zip(unique_users.to_list(), range(1, len(unique_users) + 1)))
    
    unique_items = df["work_id"].unique().sort()
    item_map = dict(zip(unique_items.to_list(), range(1, len(unique_items) + 1)))
    
    print(f"--> Vocabulary Size: {len(unique_items):,} Items")
    
    print(f"--> Saving intermediate file to {RECBOLE_INTER_FILE}...")
    df_final = df.with_columns([
        pl.col("user").replace(user_map).alias("user_id:token"),
        pl.col("work_id").replace(item_map).alias("item_id:token")
    ]).select(["user_id:token", "item_id:token"])
    
    df_final.write_csv(RECBOLE_INTER_FILE, separator="\t")
    
    print("--> Saving Intermediate BPR Maps...")
    with open(OUTPUT_DIR / "bpr_intermediate_user_map.json", "w") as f:
        json.dump(user_map, f)
        
    inv_item_map = {str(v): k for k, v in item_map.items()}
    with open(OUTPUT_DIR / "bpr_intermediate_item_map_reverse.json", "w") as f:
        json.dump(inv_item_map, f)
        
    del df, df_final, user_map, item_map, q_map, q_inter
    import gc; gc.collect()
    print("Data Preparation Complete.")

def trainer_bpr():
    print("\n=== STEP 2B: Training BPR (High Coverage) ===")
    gc.collect()
    torch.cuda.empty_cache()

    config_dict = {
        "data_path": "./dataset/",
        "dataset": DATASET_NAME,
        "field_separator": "\t",
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "load_col": {'inter': ['user_id', 'item_id']},
        
        # Much looser filters to keep more of data
        "user_inter_num_interval": "[5,inf)", 
        "item_inter_num_interval": "[2,inf)", 
        
        "embedding_size": 64,
        
        "epochs": 50,
        "train_batch_size": 4096, 
        "learner": "adam",
        "learning_rate": 0.001,
        "eval_step": 5,
        "stopping_step": 5,
        
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
        "fp16": True,
        "state": "INFO"
    }
    
    config = Config(model='BPR', config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    
    dataset = create_dataset(config)
    print(f"--> BPR Vocabulary Size: {dataset.item_num} Items")
    
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    model = BPR(config, train_data.dataset).to(config['device'])
    trainer = Trainer(config, model)
    
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    print(f"--> BPR Training Complete. Best Score: {best_valid_score}")

    # --- SAVE MAPS AND WEIGHTS ---
    print("--> Saving BPR Maps & Weights...")
    
    bpr_user_map = dataset.field2token_id['user_id']
    bpr_item_map = dataset.field2token_id['item_id']
    
    with open(OUTPUT_DIR / "bpr_user_map.json", "w") as f:
        json.dump(bpr_user_map, f)
        
    with open(OUTPUT_DIR / "bpr_item_map.json", "w") as f:
        json.dump(bpr_item_map, f)
        
    torch.save(model.state_dict(), OUTPUT_DIR / "bpr_weights.pth")
    print("✅ BPR Artifacts Saved.")

def visualize_bpr():
    
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

    min_len = min(len(epochs), len(losses), len(ndcg_scores))
    epochs = epochs[:min_len]
    losses = losses[:min_len]
    ndcg_scores = ndcg_scores[:min_len]
    recall_scores = recall_scores[:min_len]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'BPR Training Results', fontsize=16) # Changed Title

    ax1.plot(epochs, losses, 'b-o', label='Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, ndcg_scores, 'g-s', label='NDCG@10')
    ax2.plot(epochs, recall_scores, 'r-^', label='Recall@10')
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Changed filename to avoid overwriting RecVAE graph
    plot_path = OUTPUT_DIR / "bpr_training_history.png"
    plt.savefig(plot_path)
    print(f"Graph saved to: {plot_path}")
    plt.close()

def save_weights():
    # DISABLED 
    # Step 2 already handles saving BPR weights correctly.
    print("--> Step 4 skipped (BPR weights already saved in Step 2).")
    pass

if __name__ == "__main__":
    if not INPUT_INTERACTIONS.exists():
        print(f"❌ Error: Could not find {INPUT_INTERACTIONS}")
    else:
        prepare_data()
        trainer_bpr()
        visualize_bpr()
        # save_weights() -- Commented out to be safe