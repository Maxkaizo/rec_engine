
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import os

class RecommenderSystem:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.als_artifacts = None
        self.svd_model = None
        self.content_artifacts = None
        
        # Load on init
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads all .pkl artifacts from models directory"""
        print(f"Loading models from {self.models_dir}...")
        
        # ALS
        with open(os.path.join(self.models_dir, "als_artifacts.pkl"), "rb") as f:
            self.als_artifacts = pickle.load(f)
            
        # SVD
        with open(os.path.join(self.models_dir, "svd_model.pkl"), "rb") as f:
            self.svd_model = pickle.load(f)
            
        # Content
        with open(os.path.join(self.models_dir, "content_artifacts.pkl"), "rb") as f:
            self.content_artifacts = pickle.load(f)
            
        print("Models loaded successfully.")

    def get_recommendations(self, user_id, k=10, enrich=True):
        """
        Public method to get hybrid recommendations for a user.
        If enrich=True, returns a list of dicts: {'id': int, 'title': str, 'genres': str}
        If enrich=False, returns a list of MovieIDs (int).
        """
        # 1. Identify User State
        user_known = user_id in self.als_artifacts['user_map'].values()
        
        if not user_known:
            print(f"User {user_id} is unknown (Cold Start). Returning Popular items.")
            rec_ids = self._get_popular_fallback(k)
        else:
            # 2. Hybrid Candidate Generation
            candidates = self._generate_candidates(user_id)
            
            # 3. Ranking using SVD
            rec_ids = self._rank_candidates(user_id, candidates, k)
        
        # 4. Enrich with Title/Metadata if requested
        if enrich:
            return self._enrich_recommendations(rec_ids)
        
        return rec_ids

    def _enrich_recommendations(self, movie_ids):
        """
        Maps MovieIDs to Title and Genres using the movies_df artifact.
        """
        if not self.content_artifacts or 'movies_df' not in self.content_artifacts:
            # Fallback if metadata is missing
            return [{'id': mid, 'title': f"Movie {mid}", 'genres': 'Unknown'} for mid in movie_ids]
            
        movies_df = self.content_artifacts['movies_df']
        enriched = []
        
        # Optimize lookup by setting index if not already (though usually it's fast enough for top-K)
        # We assume movies_df is a DataFrame
        
        for mid in movie_ids:
            # Find row
            row = movies_df[movies_df['MovieID'] == mid]
            if not row.empty:
                title = row.iloc[0]['Title']
                genres = row.iloc[0]['Genres']
                enriched.append({'id': mid, 'title': title, 'genres': genres})
            else:
                enriched.append({'id': mid, 'title': 'Unknown', 'genres': 'Unknown'})
                
        return enriched

    def _generate_candidates(self, user_id):
        """
        Stage 1: Retrieval
        Combines ALS Collaborative Filtering + Content-Based (TF-IDF)
        """
        candidates = set()
        
        # --- A. ALS Candidates ---
        # Map Real UserID -> Internal Index
        # Note: In train.py: "user_map": dict(enumerate(full_users.categories)) -> {0: 1, 1: 2...} (Internal -> Real)
        
        # We need Real -> Internal to query the model
        real_to_internal_user = {v: k for k, v in self.als_artifacts['user_map'].items()}
        
        if user_id in real_to_internal_user:
            u_idx = real_to_internal_user[user_id]
            user_items = self.als_artifacts['user_item_matrix']
            
            # recommend() returns Internal Item Indices
            ids, scores = self.als_artifacts['model'].recommend(
                u_idx, 
                user_items[u_idx], 
                N=50, 
                filter_already_liked_items=True
            )
            
            # Convert Internal Item Index -> Real MovieID
            internal_to_real_movie = self.als_artifacts['movie_map'] # {0: 1, 1: 2...}
            als_candidates = [internal_to_real_movie[i] for i in ids]
            candidates.update(als_candidates)
            
        # --- B. Content Candidates ---
        # For simplicity in this version, we'll skip complex content logic here 
        # and rely mainly on ALS, but the structure is ready for it.
            
        return list(candidates)

    def _rank_candidates(self, user_id, candidates, k):
        """
        Stage 2: Ranking
        Scores candidates using SVD (Matrix Factorization)
        """
        scored_items = []
        for movie_id in candidates:
            # Predict Rating
            prediction = self.svd_model.predict(user_id, movie_id)
            est_rating = prediction.est
            scored_items.append((movie_id, est_rating))
            
        # Sort by predicted rating
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        return [item[0] for item in scored_items[:k]]

    def _get_popular_fallback(self, k):
        """
        Returns top popular items computed during training.
        """
        if self.content_artifacts and 'popular_movies' in self.content_artifacts:
            # Return top K from the pre-computed list
            return self.content_artifacts['popular_movies'][:k]
        
        # Fallback if artifact is missing (safety net)
        return [2858, 260, 1196, 1210, 480, 2028, 589, 2571, 1270, 593]


