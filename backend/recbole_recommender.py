import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from metadata_loader import id_to_metadata

# Define paths
MODEL_DATA_DIR = Path(__file__).parent / 'data/processed'

# ==========================================
# 1. Model Architecture (Must match training)
# ==========================================
class RecVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=600, latent_dim=200):
        super(RecVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, rating_matrix):
        x = torch.nn.functional.normalize(rating_matrix, p=2, dim=1)
        h = self.encoder(x)
        mu, _ = torch.chunk(h, 2, dim=1)
        z = mu 
        return self.decoder(z)

# ==========================================
# 2. The Inference Engine
# ==========================================
class RecommenderEngine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = None
        self.is_ready = False
        
        # Maps
        self.recbole_to_work = {}
        self.work_to_recbole = {}
        self.sheet_to_work = {}
        self.work_to_sheets = {}
        
        try:
            self._load_artifacts()
        except Exception as e:
            print(f"⚠️ Warning: AI Recommender failed to load: {e}")
            print("Recommendations will be empty.")

    def _load_artifacts(self):
        print(f"Loading AI Model from {MODEL_DATA_DIR}...")
        
        # 1. Load Maps
        with open(MODEL_DATA_DIR / "item_map_reverse.json") as f:
            self.recbole_to_work = json.load(f)
            self.work_to_recbole = {v: int(k) for k, v in self.recbole_to_work.items()}
            
        with open(MODEL_DATA_DIR / "sheet_to_work.json") as f:
            self.sheet_to_work = json.load(f)
            
        with open(MODEL_DATA_DIR / "work_to_sheets.json") as f:
            self.work_to_sheets = json.load(f)

        # 2. Initialize Model
        # Input dim is vocabulary size + 1 (for padding)
        input_dim = len(self.recbole_to_work) + 1 
        self.model = RecVAE(input_dim)
        
        # 3. Load Weights
        weight_path = MODEL_DATA_DIR / "recvae_weights.pth"
        checkpoint = torch.load(weight_path, map_location=self.device)
        
        # Handle RecBole checkpoint wrapper
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.is_ready = True
        print("✅ RecVAE Model Loaded Successfully.")

    def recommend(self, liked_sheet_ids, top_k=10):
        if not self.is_ready or not liked_sheet_ids:
            return []

        # A. Convert Sheet IDs -> Input Vector Indices
        input_indices = []
        for sheet_id in liked_sheet_ids:
            # Sheet ID -> Work ID
            if sheet_id in self.sheet_to_work:
                work_id = str(self.sheet_to_work[sheet_id])
                # Work ID -> RecBole Index
                if work_id in self.work_to_recbole:
                    input_indices.append(self.work_to_recbole[work_id])
        
        input_indices = list(set(input_indices))
        if not input_indices:
            return [] # Cold start

        # B. Build Vector
        input_dim = len(self.recbole_to_work) + 1
        input_vector = torch.zeros(input_dim)
        input_vector[input_indices] = 1.0
        
        # C. Predict
        with torch.no_grad():
            scores = self.model(input_vector.unsqueeze(0))
            
            # Mask out items user already liked (so we don't recommend them again)
            scores[0, input_indices] = -float('inf')
            scores[0, 0] = -float('inf') # Mask padding

            _, top_indices = torch.topk(scores, top_k)
            top_indices = top_indices.squeeze().tolist()

        # D. Convert Indices back to Sheet IDs
        recommended_sheets = []
        for idx in top_indices:
            idx_str = str(idx)
            if idx_str in self.recbole_to_work:
                work_id = str(self.recbole_to_work[idx_str])
                
                if work_id in self.work_to_sheets:
                    sheets = self.work_to_sheets[work_id]
                    # Pick the first sheet ID for this work
                    best_sheet = sheets[0] if isinstance(sheets, list) else sheets
                    recommended_sheets.append(best_sheet)
                    
        return recommended_sheets

# Initialize Global Engine
rec_engine = RecommenderEngine()

# ==========================================
# 3. Public API Function
# ==========================================
def get_recommendations_by_song_ids(song_ids, k=10):
    """
    Args:
        song_ids: List of MuseScore song IDs (strings)
        k: Number of recommendations
    Returns:
        List of full metadata objects for recommended songs
    """
    # 1. Get Recommended IDs from AI
    rec_ids = rec_engine.recommend(song_ids, top_k=k)
    
    # 2. Hydrate with Metadata (Title, Artist, etc.)
    results = []
    for rid in rec_ids:
        # Check if ID exists in your metadata loader
        if rid in id_to_metadata:
            # Return the full object so the frontend can render cards
            results.append(id_to_metadata[rid])
            
    return results