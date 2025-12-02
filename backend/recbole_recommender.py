import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path

# Define paths
MODEL_DATA_DIR = Path(__file__).parent.resolve()


# ==========================================
# RecVAE Model Architecture (RecBole)
# ==========================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=600, latent_dim=200):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = torch.tanh(self.ln1(self.fc1(x)))
        h = torch.tanh(self.ln2(self.fc2(h)))
        h = torch.tanh(self.ln3(self.fc3(h)))
        h = torch.tanh(self.ln4(self.fc4(h)))
        h = torch.tanh(self.ln5(self.fc5(h)))
        return self.fc_mu(h), self.fc_logvar(h)


class Prior(nn.Module):
    def __init__(self, input_dim, hidden_dim=600, latent_dim=200):
        super(Prior, self).__init__()
        self.encoder_old = Encoder(input_dim, hidden_dim, latent_dim)
        self.mu_prior = nn.Parameter(torch.zeros(1, latent_dim))
        self.logvar_prior = nn.Parameter(torch.zeros(1, latent_dim))
        self.logvar_uniform_prior = nn.Parameter(torch.zeros(1, latent_dim))


class RecVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=600, latent_dim=200):
        super(RecVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.prior = Prior(input_dim, hidden_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, rating_matrix):
        x = torch.nn.functional.normalize(rating_matrix, p=2, dim=1)
        mu, _ = self.encoder(x)
        return self.decoder(mu)


print("Pre-loading PyTorch models...")
_preloaded_recvae_model = None
_preloaded_bpr_embeddings = None
_preload_error = None

try:
    # Load and create RecVAE model
    recvae_ckpt = torch.load(
        MODEL_DATA_DIR / "models" / "recvae_weights.pth", 
        map_location='cpu', 
        weights_only=False
    )
    state_dict = recvae_ckpt['state_dict'] if 'state_dict' in recvae_ckpt else recvae_ckpt
    
    # Infer dimensions from saved weights
    # decoder.weight shape is [output_dim, input_dim] = [num_items, latent_dim]
    decoder_weight = state_dict['decoder.weight']
    input_dim = decoder_weight.shape[0]     # Number of items (84482)
    latent_dim = decoder_weight.shape[1]    # Latent dimension (200)
    
    # Get hidden_dim from encoder layers
    # encoder.fc1.weight shape is [hidden_dim, input_dim]
    hidden_dim = state_dict['encoder.fc1.weight'].shape[0]  # 600
    
    print(f"  Detected dimensions: input_dim={input_dim}, hidden_dim={hidden_dim}, latent_dim={latent_dim}")
    
    _preloaded_recvae_model = RecVAE(input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    _preloaded_recvae_model.load_state_dict(state_dict)
    _preloaded_recvae_model.eval()
    del recvae_ckpt, state_dict
    print(f"  RecVAE model pre-loaded ({input_dim} items, hidden={hidden_dim}, latent={latent_dim})")
except Exception as e:
    print(f"  RecVAE pre-load failed: {e}")
    _preload_error = str(e)


try:
    # Load BPR embeddings
    bpr_ckpt = torch.load(
        MODEL_DATA_DIR / "models" / "bpr_weights.pth", 
        map_location='cpu', 
        weights_only=False
    )
    _preloaded_bpr_embeddings = bpr_ckpt['item_embedding.weight']
    _preloaded_bpr_embeddings = torch.nn.functional.normalize(_preloaded_bpr_embeddings, p=2, dim=1)
    del bpr_ckpt
    print(f"  BPR embeddings pre-loaded ({_preloaded_bpr_embeddings.shape[0]} items)")
except Exception as e:
    print(f"  BPR pre-load failed: {e}")
    _preload_error = str(e)

from metadata_loader import id_to_metadata

# Import embedding search for fallback similarity matching
from search_embedding import find_similar_sheet_with_mapping


# ==========================================
# Hybrid Recommender Engine
# ==========================================
class HybridRecommenderEngine:
    """
    A hybrid recommender that combines RecVAE and BPR models.
    
    Strategy:
    - RecVAE: Trained on ~84K most popular works. Best for users with 
      interactions in the popular item set.
    - BPR: Trained on ~612K works. Uses item embeddings for similarity-based 
      recommendations. Good fallback for less popular items.
    
    For new users (cold start), we:
    1. Take their favorited songs or a single song as input
    2. Map sheet IDs -> work IDs
    3. Try RecVAE first if items are in its vocabulary
    4. Fall back to BPR embedding similarity otherwise
    5. Blend results if items span both models
    """
    
    def __init__(self):
        self.device = torch.device("cpu")
        
        # Models
        self.recvae_model = None
        self.bpr_item_embeddings = None
        
        # Mapping: sheet_id <-> work_id <-> model_idx
        self.sheet_to_work = {}      # sheet_id -> work_id
        self.work_to_sheets = {}     # work_id -> [sheet_ids]
        
        # RecVAE mappings (covers ~84K popular works)
        self.recvae_work_to_idx = {} # work_id -> recvae_idx
        self.recvae_idx_to_work = {} # recvae_idx -> work_id
        self.recvae_input_dim = 0
        
        # BPR mappings (covers ~612K works)
        self.bpr_work_to_idx = {}    # work_id -> bpr_idx
        self.bpr_idx_to_work = {}    # bpr_idx -> work_id
        
        # Status flags
        self.recvae_ready = False
        self.bpr_ready = False
        self._initialized = False
        
        # Valid work IDs (ones that have sheet mappings)
        self.valid_work_ids = set()
        
        # Precomputed masks for efficient filtering
        self.recvae_valid_mask = None
        self.bpr_valid_mask = None
        
        # Lazy loading - only load mappings at init, load models on first use
        self._load_mappings_only()
    
    def _ensure_models_loaded(self):
        """Lazy load models on first use to avoid memory issues at startup."""
        if self._initialized:
            return
        self._initialized = True
        
        # Load RecVAE
        try:
            self._load_recvae()
        except Exception as e:
            print(f"  RecVAE failed to load: {e}")
        
        # Load BPR
        try:
            self._load_bpr()
        except Exception as e:
            print(f"  BPR failed to load: {e}")
        
        if not self.recvae_ready and not self.bpr_ready:
            print("Warning: No recommendation models loaded. Recommendations will be empty.")
        else:
            print("Hybrid Recommender Models Loaded!")
    
    def _load_mappings_only(self):
        """Load only the ID mappings at startup (lightweight)."""
        print(f"Loading Hybrid Recommender mappings from {MODEL_DATA_DIR}...")
        
        # Load sheet <-> work mappings
        try:
            with open(MODEL_DATA_DIR / "api_lookups" / "sheet_to_work.json") as f:
                self.sheet_to_work = {str(k): int(v) for k, v in json.load(f).items()}
            with open(MODEL_DATA_DIR / "api_lookups" / "work_to_sheets.json") as f:
                self.work_to_sheets = {str(k): v for k, v in json.load(f).items()}
            # Track which work IDs are valid (have sheet mappings)
            self.valid_work_ids = set(int(w) for w in self.work_to_sheets.keys())
            print(f"  Loaded sheet/work mappings: {len(self.sheet_to_work)} sheets -> {len(self.work_to_sheets)} works")
        except Exception as e:
            print(f"  Failed to load sheet/work mappings: {e}")
        
        # Load item ID mappings (lightweight JSON files)
        try:
            with open(MODEL_DATA_DIR / "models" / "item_map_reverse.json") as f:
                idx_to_work = json.load(f)
                self.recvae_idx_to_work = {int(k): int(v) for k, v in idx_to_work.items()}
                self.recvae_work_to_idx = {v: k for k, v in self.recvae_idx_to_work.items()}
            print(f"  Loaded RecVAE mappings: {len(self.recvae_work_to_idx)} works")
        except Exception as e:
            print(f"  Failed to load RecVAE mappings: {e}")
            
        try:
            with open(MODEL_DATA_DIR / "models" / "bpr_item_map.json") as f:
                bpr_map = json.load(f)
                self.bpr_work_to_idx = {int(k): int(v) for k, v in bpr_map.items() if k != '[PAD]'}
                self.bpr_idx_to_work = {v: k for k, v in self.bpr_work_to_idx.items()}
            print(f"  Loaded BPR mappings: {len(self.bpr_work_to_idx)} works")
        except Exception as e:
            print(f"  Failed to load BPR mappings: {e}")
        
        print("Hybrid Recommender Ready (models will load on first use)")
    
    def _load_recvae(self):
        """Initialize RecVAE from pre-loaded model and create valid mask."""
        global _preloaded_recvae_model
        
        print("  Initializing RecVAE...")
        
        if _preloaded_recvae_model is None:
            raise RuntimeError("RecVAE model was not pre-loaded successfully")
        
        self.recvae_model = _preloaded_recvae_model
        self.recvae_input_dim = self.recvae_model.decoder.weight.shape[1]
        
        # Create mask for valid items (ones with sheet mappings and within model bounds)
        self.recvae_valid_mask = torch.zeros(self.recvae_input_dim, dtype=torch.bool)
        valid_count = 0
        for work_id in self.valid_work_ids:
            if work_id in self.recvae_work_to_idx:
                idx = self.recvae_work_to_idx[work_id]
                if idx < self.recvae_input_dim:
                    self.recvae_valid_mask[idx] = True
                    valid_count += 1
        
        self.recvae_ready = True
        print(f"  RecVAE loaded: {self.recvae_input_dim} items, {valid_count} with sheet mappings")
    
    def _load_bpr(self):
        """Load BPR item embeddings and create valid mask."""
        global _preloaded_bpr_embeddings
        
        print("  Initializing BPR...")
        
        if _preloaded_bpr_embeddings is None:
            raise RuntimeError("BPR embeddings were not pre-loaded successfully")
        
        self.bpr_item_embeddings = _preloaded_bpr_embeddings
        
        # Create mask for valid items (ones with sheet mappings)
        num_items = self.bpr_item_embeddings.shape[0]
        self.bpr_valid_mask = torch.zeros(num_items, dtype=torch.bool)
        valid_count = 0
        for work_id in self.valid_work_ids:
            if work_id in self.bpr_work_to_idx:
                self.bpr_valid_mask[self.bpr_work_to_idx[work_id]] = True
                valid_count += 1
        
        self.bpr_ready = True
        print(f"  BPR loaded: {num_items} items, {valid_count} with sheet mappings, embedding dim {self.bpr_item_embeddings.shape[1]}")
    
    def _sheet_ids_to_work_ids(self, sheet_ids, use_fallback=True):
        """
        Convert sheet IDs to work IDs.
        Ensures the mapped work IDs are actually KNOWN to the model.
        """
        work_ids = set()
        unmapped_sheets = []
        
        # Pre-calculate known IDs to validate mappings
        valid_model_ids = set()
        if self.bpr_work_to_idx:
            valid_model_ids.update(self.bpr_work_to_idx.keys())
        if self.recvae_work_to_idx:
            valid_model_ids.update(self.recvae_work_to_idx.keys())
        
        print(f"\n[Mapping] Processing {len(sheet_ids)} sheet IDs...")  
        print(f"[Mapping] Valid model IDs available: {len(valid_model_ids)}")  
        
        for sid in sheet_ids:
            sid_str = str(sid)
            is_mapped = False
            
            # Check if we have a direct mapping
            if sid_str in self.sheet_to_work:
                mapped_id = self.sheet_to_work[sid_str]
                
                if mapped_id in valid_model_ids:
                    work_ids.add(mapped_id)
                    is_mapped = True
                    print(f"  Sheet {sid} -> work_id {mapped_id} (direct mapping)")  
                else:
                    print(f"  Sheet {sid} maps to {mapped_id}, but ID is unknown to model. Using fallback.")
            else:
                print(f"  Sheet {sid} has no direct mapping") 
            
            if not is_mapped:
                unmapped_sheets.append(sid_str)
        
        # Use embedding similarity fallback for unmapped sheets
        if use_fallback and unmapped_sheets:
            print(f"  → Attempting fallback search for {len(unmapped_sheets)} sheets...")
            
            for sid in unmapped_sheets:
                try:
                    fallback_work_id, score = find_similar_sheet_with_mapping(
                        sid,
                        self.sheet_to_work,
                        k=50,  
                        allowed_work_ids=valid_model_ids  
                    )
                    if fallback_work_id is not None and score >= 0.25:  
                        work_ids.add(fallback_work_id)
                        print(f"    Fallback: Sheet {sid} -> work_id={fallback_work_id} (score={score:.3f})")
                    else:
                        print(f"    Fallback failed: No valid match found for Sheet {sid} (best score={score:.3f})") 
                except Exception as e:
                    print(f"    Fallback error for sheet {sid}: {e}")
                    import traceback
                    traceback.print_exc()  
        
        print(f"[Mapping] Final result: {len(work_ids)} valid work IDs\n") 
        return list(work_ids)
    
    def _work_ids_to_sheet_ids(self, work_ids, exclude_sheets=None):
        """
        Convert work IDs back to sheet IDs.
        Returns best sheet for each work (first in list).
        """
        exclude_sheets = set(str(s) for s in (exclude_sheets or []))
        sheet_ids = []
        
        print(f"\n[W2S] Converting {len(work_ids)} work_ids to sheet_ids...")
        print(f"[W2S] Excluding {len(exclude_sheets)} sheets: {list(exclude_sheets)[:5]}")
        
        for wid in work_ids:
            wid_str = str(wid)
            if wid_str in self.work_to_sheets:
                sheets = self.work_to_sheets[wid_str]
                if isinstance(sheets, list):
                    # Find first sheet not in exclude list
                    found = False
                    for s in sheets:
                        if str(s) not in exclude_sheets:
                            sheet_ids.append(str(s))
                            found = True
                            break
                    if not found:
                        print(f"  work_id {wid} has sheets {sheets} but all are excluded")
                else:
                    if str(sheets) not in exclude_sheets:
                        sheet_ids.append(str(sheets))
                    else:
                        print(f"  work_id {wid} sheet {sheets} is excluded")
            else:
                print(f"  work_id {wid} not found in work_to_sheets mapping")
        
        print(f"[W2S] Result: {len(sheet_ids)} sheet_ids")
        return sheet_ids
    
    def _recommend_recvae(self, work_ids, top_k, exclude_work_ids):
        """
        Get recommendations using RecVAE.
        Returns (recommended_work_ids, scores) or ([], []) if not applicable.
        """
        if not self.recvae_ready:
            return [], []
        
        # Map work_ids to RecVAE indices
        input_indices = []
        for wid in work_ids:
            if wid in self.recvae_work_to_idx:
                idx = self.recvae_work_to_idx[wid]
                if idx < self.recvae_input_dim:  # Ensure within model bounds
                    input_indices.append(idx)
        
        if not input_indices:
            return [], []
        
        # Build input vector
        input_vector = torch.zeros(self.recvae_input_dim)
        input_vector[input_indices] = 1.0
        
        # Get predictions
        with torch.no_grad():
            scores = self.recvae_model(input_vector.unsqueeze(0)).squeeze(0)
            
            # Mask out invalid items (no sheet mappings)
            scores[~self.recvae_valid_mask] = -float('inf')
            
            # Mask out input items and excluded items
            scores[input_indices] = -float('inf')
            scores[0] = -float('inf')  # Mask padding index
            
            for wid in exclude_work_ids:
                if wid in self.recvae_work_to_idx:
                    idx = self.recvae_work_to_idx[wid]
                    if idx < self.recvae_input_dim:
                        scores[idx] = -float('inf')
            
            # Get top-k
            top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
        
        # Convert indices back to work IDs
        rec_work_ids = []
        rec_scores = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            if idx in self.recvae_idx_to_work and score > -float('inf'):
                rec_work_ids.append(self.recvae_idx_to_work[idx])
                rec_scores.append(score)
        
        return rec_work_ids, rec_scores
    
    def _recommend_bpr(self, work_ids, top_k, exclude_work_ids):
        """
        Get recommendations using BPR item embeddings (similarity-based).
        Computes average embedding of input items and finds most similar items.
        Only returns work IDs that have sheet mappings.
        Returns (recommended_work_ids, scores) or ([], []) if not applicable.
        """
        if not self.bpr_ready:
            return [], []
        
        # Map work_ids to BPR indices
        input_indices = []
        for wid in work_ids:
            if wid in self.bpr_work_to_idx:
                input_indices.append(self.bpr_work_to_idx[wid])
        
        if not input_indices:
            return [], []
        
        # Compute average embedding of input items
        input_embeddings = self.bpr_item_embeddings[input_indices]
        query_embedding = input_embeddings.mean(dim=0, keepdim=True)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        
        # Compute cosine similarity with all items
        similarities = torch.mm(query_embedding, self.bpr_item_embeddings.t()).squeeze(0)
        
        # Mask out invalid items (no sheet mappings)
        similarities[~self.bpr_valid_mask] = -float('inf')
        
        # Mask out input items and excluded items
        for idx in input_indices:
            similarities[idx] = -float('inf')
        similarities[0] = -float('inf')  # Mask padding
        
        for wid in exclude_work_ids:
            if wid in self.bpr_work_to_idx:
                similarities[self.bpr_work_to_idx[wid]] = -float('inf')
        
        # Get top-k
        top_scores, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
        
        # Convert indices back to work IDs
        rec_work_ids = []
        rec_scores = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            if idx in self.bpr_idx_to_work and score > -float('inf'):
                rec_work_ids.append(self.bpr_idx_to_work[idx])
                rec_scores.append(score)
        
        return rec_work_ids, rec_scores
    
    def recommend(self, sheet_ids, top_k=10):
        """
        Get recommendations for a list of sheet IDs.
        
        Args:
            sheet_ids: List of MuseScore sheet IDs (strings or ints)
            top_k: Number of recommendations to return
            
        Returns:
            List of recommended sheet IDs (strings)
        """
        if not sheet_ids:
            return []
        
        # Lazy load models on first use
        self._ensure_models_loaded()
        
        sheet_ids = [str(s) for s in sheet_ids]
        print(f"\n[Recommend] Starting recommendation for sheet_ids: {sheet_ids}")
        
        # Convert to work IDs
        work_ids = self._sheet_ids_to_work_ids(sheet_ids)
        if not work_ids:
            print("  No valid work IDs found for inputs (all unknown to model).")
            # Use actual popular items from the valid set
            if self.bpr_ready and self.valid_work_ids:
                print("  Using popular BPR items as fallback")
                # Get intersection of BPR items and valid work IDs
                popular_candidates = [wid for wid in self.bpr_work_to_idx.keys() 
                                    if wid in self.valid_work_ids][:top_k * 2]
                if popular_candidates:
                    fallback_sheets = self._work_ids_to_sheet_ids(popular_candidates[:top_k], exclude_sheets=sheet_ids)
                    print(f"  Returning {len(fallback_sheets)} fallback recommendations")
                    return fallback_sheets
            return []
        
        print(f"[Recommend] Input work_ids: {work_ids}")
        exclude_work_ids = set(work_ids)
        
        # Categorize input work IDs by model coverage
        recvae_work_ids = [w for w in work_ids if w in self.recvae_work_to_idx]
        bpr_only_work_ids = [w for w in work_ids if w not in self.recvae_work_to_idx and w in self.bpr_work_to_idx]
        
        print(f"[Recommend] RecVAE coverage: {len(recvae_work_ids)}/{len(work_ids)} works")
        print(f"[Recommend] BPR-only coverage: {len(bpr_only_work_ids)}/{len(work_ids)} works")
        
        # Check if work_ids are actually in either model
        if not recvae_work_ids and not bpr_only_work_ids:
            print(f"  ERROR: work_ids {work_ids} not found in RecVAE OR BPR vocabularies!")
            print(f"  RecVAE has {len(self.recvae_work_to_idx)} works")
            print(f"  BPR has {len(self.bpr_work_to_idx)} works")
            print(f"  Sample RecVAE IDs: {list(self.recvae_work_to_idx.keys())[:10]}")
            print(f"  Sample BPR IDs: {list(self.bpr_work_to_idx.keys())[:10]}")
            return []
        
        all_recommendations = []
        
        # Get RecVAE recommendations if we have relevant items
        if recvae_work_ids:
            print(f"[Recommend] Calling RecVAE with {len(recvae_work_ids)} works...")
            recvae_recs, recvae_scores = self._recommend_recvae(
                recvae_work_ids, 
                top_k=top_k * 2,
                exclude_work_ids=exclude_work_ids
            )
            print(f"  RecVAE returned {len(recvae_recs)} recommendations")
            if len(recvae_recs) > 0:
                print(f"  Sample RecVAE recs: {recvae_recs[:3]}, scores: {recvae_scores[:3]}")
            for wid, score in zip(recvae_recs, recvae_scores):
                all_recommendations.append((wid, score, 'recvae'))
        
        # Get BPR recommendations
        if self.bpr_ready:
            # Use all work IDs that BPR knows about
            bpr_input_ids = [w for w in work_ids if w in self.bpr_work_to_idx]
            if bpr_input_ids:
                print(f"[Recommend] Calling BPR with {len(bpr_input_ids)} works...")
                bpr_recs, bpr_scores = self._recommend_bpr(
                    bpr_input_ids,
                    top_k=top_k * 2,
                    exclude_work_ids=exclude_work_ids
                )
                print(f"  BPR returned {len(bpr_recs)} recommendations")
                if len(bpr_recs) > 0:
                    print(f"  Sample BPR recs: {bpr_recs[:3]}, scores: {bpr_scores[:3]}")
                for wid, score in zip(bpr_recs, bpr_scores):
                    all_recommendations.append((wid, score, 'bpr'))
        
        print(f"[Recommend] Total recommendations before dedup: {len(all_recommendations)}")
        
        # Deduplicate and rank
        seen_work_ids = set()
        final_work_ids = []
        
        # Sort by source priority then score
        source_priority = {'recvae': 0, 'bpr': 1}
        all_recommendations.sort(key=lambda x: (source_priority[x[2]], -x[1]))
        
        for wid, score, source in all_recommendations:
            if wid not in seen_work_ids:
                seen_work_ids.add(wid)
                final_work_ids.append(wid)
                if len(final_work_ids) >= top_k:
                    break
        
        print(f"[Recommend] Final work_ids after dedup: {len(final_work_ids)}")
        if final_work_ids:
            print(f"  Sample final work_ids: {final_work_ids[:5]}")
        
        # Convert back to sheet IDs
        print(f"[Recommend] Converting {len(final_work_ids)} work_ids to sheet_ids...")
        recommended_sheets = self._work_ids_to_sheet_ids(
            final_work_ids, 
            exclude_sheets=sheet_ids
        )
        
        print(f"[Recommend] Final result: {len(recommended_sheets)} sheet_ids")
        if recommended_sheets:
            print(f"  Sample sheet_ids: {recommended_sheets[:5]}")
        else:
            print(f"  WARNING: work_ids converted to 0 sheet_ids!")
            # Debug: check if work_ids have mappings
            for wid in final_work_ids[:5]:
                wid_str = str(wid)
                if wid_str in self.work_to_sheets:
                    print(f"    work_id {wid} HAS mapping: {self.work_to_sheets[wid_str]}")
                else:
                    print(f"    work_id {wid} has NO mapping in work_to_sheets")
        
        return recommended_sheets[:top_k]


# Initialize Global Engine
rec_engine = HybridRecommenderEngine()

# ==========================================
# Public API Functions
# ==========================================
def get_recommendations_by_song_ids(song_ids, k=10):
    """
    Get recommendations based on a list of liked/favorited song IDs.
    
    Args:
        song_ids: List of MuseScore sheet IDs (strings)
        k: Number of recommendations to return
        
    Returns:
        List of full metadata objects for recommended songs
    """
    if not song_ids:
        return []
    
    # Get recommended sheet IDs from hybrid model
    rec_sheet_ids = rec_engine.recommend(song_ids, top_k=k)
    
    results = []
    for sheet_id in rec_sheet_ids:
        if sheet_id in id_to_metadata:
            results.append(id_to_metadata[sheet_id])
    
    return results


def get_similar_songs(song_id, k=10):
    """
    Find songs similar to a single song.
    Convenience wrapper for single-song recommendations.
    
    Args:
        song_id: A single MuseScore sheet ID (string)
        k: Number of similar songs to return
        
    Returns:
        List of full metadata objects for similar songs
    """
    return get_recommendations_by_song_ids([song_id], k=k)


# Test if run directly
if __name__ == "__main__":
    print("\n--- Testing Hybrid Recommender ---")
    
    # Get some sample sheet IDs from metadata
    sample_sheets = list(id_to_metadata.keys())[:5]
    print(f"Sample input sheets: {sample_sheets}")
    
    # Test recommendations
    recs = rec_engine.recommend(sample_sheets, top_k=5)
    print(f"Recommendations: {recs}")
    
    # Test with metadata hydration
    full_recs = get_recommendations_by_song_ids(sample_sheets, k=5)
    print(f"Full recommendations: {len(full_recs)} items")
    for r in full_recs[:3]:
        print(f"  - {r.get('title', 'Unknown')} by {r.get('authorUserId', 'Unknown')}")