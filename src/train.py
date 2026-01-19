
import pandas as pd
import scipy.sparse as sparse
import implicit
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Configuration
SEED = 42
RATINGS_FILE = "data/ml-1m/ratings.dat"
MOVIES_FILE = "data/ml-1m/movies.dat"
MODELS_DIR = "models"

# Hyperparameters (Tuned in 02_Model_Training.ipynb)
ALS_FACTORS = 32
ALS_REGULARIZATION = 0.05
ALS_ITERATIONS = 20

SVD_N_FACTORS = 50 
SVD_LR_ALL = 0.01
SVD_EPOCHS = 20

def main():
    print("--- Starting Production Training Job ---")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load Data
    print("Loading data...")
    ratings_cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    ratings = pd.read_csv(RATINGS_FILE, sep='::', header=None, names=ratings_cols, engine='python', encoding='latin-1')
    
    movies_cols = ['MovieID', 'Title', 'Genres']
    movies = pd.read_csv(MOVIES_FILE, sep='::', header=None, names=movies_cols, engine='python', encoding='latin-1')

    all_users = ratings['UserID'].unique()
    all_movies = ratings['MovieID'].unique()

    # 2. Content-Based (TF-IDF)
    print("Computing TF-IDF matrices...")
    movies['genres_str'] = movies['Genres'].str.replace('|', ' ', regex=False)
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b[A-Za-z-]+\b")
    tfidf_matrix = tfidf.fit_transform(movies['genres_str'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 3. ALS Training
    print(f"Training ALS Model (factors={ALS_FACTORS}, reg={ALS_REGULARIZATION})...")
    # Categoricals for consistent mapping
    full_users = pd.Categorical(ratings['UserID'], categories=all_users)
    full_movies = pd.Categorical(ratings['MovieID'], categories=all_movies)

    # Matrix for Implicit (Item x User)
    item_user_full = sparse.csr_matrix(
        (ratings['Rating'].astype(float), (full_movies.codes, full_users.codes)),
        shape=(len(all_movies), len(all_users))
    )
    
    # Transpose to User x Item for fit() and recommend()
    user_item_full = item_user_full.T.tocsr() # Optimization: User x Item

    als_model = implicit.als.AlternatingLeastSquares(
        factors=ALS_FACTORS, 
        regularization=ALS_REGULARIZATION, 
        iterations=ALS_ITERATIONS, 
        random_state=SEED
    )
    als_model.fit(user_item_full) 

    # 4. SVD Training
    print(f"Training SVD Model (n_factors={SVD_N_FACTORS}, lr={SVD_LR_ALL})...")
    reader = Reader(rating_scale=(1, 5))
    data_full = Dataset.load_from_df(ratings[['UserID', 'MovieID', 'Rating']], reader)
    trainset_full = data_full.build_full_trainset()
    
    svd_model = SVD(
        n_factors=SVD_N_FACTORS, 
        lr_all=SVD_LR_ALL, 
        n_epochs=SVD_EPOCHS, 
        random_state=SEED
    )
    svd_model.fit(trainset_full)

    # 5. Save Artifacts
    print("Saving artifacts to models/...")
    
    # Mappings
    user_map = dict(enumerate(full_users.categories))
    movie_map = dict(enumerate(full_movies.categories))
    
    # ALS Artifacts
    als_artifacts = {
        "model": als_model,
        "user_item_matrix": user_item_full,
        "user_inv_map": {v: k for k, v in user_map.items()},
        "movie_inv_map": {v: k for k, v in movie_map.items()},
        "user_map": user_map,
        "movie_map": movie_map
    }
    with open(os.path.join(MODELS_DIR, "als_artifacts.pkl"), "wb") as f:
        pickle.dump(als_artifacts, f)

    # SVD Model
    with open(os.path.join(MODELS_DIR, "svd_model.pkl"), "wb") as f:
        pickle.dump(svd_model, f)

    # Content Artifacts
    content_artifacts = {
        "tfidf_matrix": tfidf_matrix,
        "tfidf_vectorizer": tfidf,
        "cosine_sim_matrix": cosine_sim,
        "movies_df": movies[['MovieID', 'Title', 'Genres']]
    }
    with open(os.path.join(MODELS_DIR, "content_artifacts.pkl"), "wb") as f:
        pickle.dump(content_artifacts, f)

    print("Training Complete. All models saved.")

if __name__ == "__main__":
    main()
