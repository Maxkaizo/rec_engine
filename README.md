
# MovieLens 1M Hybrid Recommender System

**Capstone Project - Machine Learning Zoomcamp**

## 1. Problem Description

Content discovery is a significant challenge in modern streaming platforms. With thousands of movies available, users often struggle to find content that matches their specific tastes.

**Goal**: Build a robust, hybrid recommendation system using the **MovieLens 1M** dataset. The system aims to provide personalized top-N movie recommendations for existing users and handle cold-start scenarios using popularity-based fallbacks.

**Solution Architecture**:
We implemented a **Two-Stage Pipeline** to balance efficiency and accuracy:
1.  **Candidate Generation (Retrieval)**:
    *   **Collaborative Filtering**: Using **ALS (Alternating Least Squares)** from the `implicit` library to retrieve ~100 candidate movies based on user interaction history.
    *   **Content-Based Filtering**: Using **TF-IDF** on movie genres to find items similar to a user's highly-rated movies (hybridization).
2.  **Ranking**:
    *   **Matrix Factorization**: Using **SVD (Singular Value Decomposition)** from `scikit-surprise` to predict explicit ratings (1-5 stars) for the retrieved candidates and re-rank them.

## 2. Project Structure

```
rec_engine/
├── data/                   # Dataset directory (MovieLens 1M)
├── docs/                   # Documentation and design notes
├── models/                 # Saved model artifacts (.pkl)
├── notebooks/              # Jupyter notebooks for EDA and Prototyping
│   ├── 01_EDA.ipynb        # Exploratory Data Analysis
│   └── 02_Model_Training.ipynb # Model tuning and pipeline verification
├── src/                    # Source code for production
│   └── train.py            # Production training script
├── pyproject.toml          # Dependencies (managed by uv)
└── README.md               # Project documentation
```

## 3. Setup and Installation

This project uses **uv** for fast and reliable dependency management.

### Prerequisites
*   Linux/macOS (or WSL on Windows)
*   Python 3.12+
*   `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Installation Steps
1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd rec_engine
    ```

2.  **Install Dependencies**:
    ```bash
    uv sync
    ```

3.  **Activate Virtual Environment**:
    ```bash
    source .venv/bin/activate
    ```

### Data Preparation
1.  Download the **MovieLens 1M** dataset from [GroupLens](https://grouplens.org/datasets/movielens/1m/).
2.  Extract the contents (`ratings.dat`, `movies.dat`, `users.dat`) into `rec_engine/data/ml-1m/`.

## 4. Exploratory Data Analysis (EDA)

We conducted an extensive analysis in `notebooks/01_EDA.ipynb`, covering:
*   **Rating Distribution**: Confirmed a long-tail distribution (popularity bias).
*   **User Activity**: Analyzed the "cold-start" threshold; users typically have >20 ratings.
*   **Genre Analysis**: Visualized genre co-occurrences and validated TF-IDF similarity using cosine similarity on movie genres.
*   **Feasibility Check**: Confirmed >93% of users have enough high-rated (4+) movies to support content-based candidate generation.

## 5. Model Training & Strategy

The training process (`src/train.py` / `notebooks/02_Model_Training.ipynb`) follows a rigorous **3-way split** strategy:

1.  **Split**:
    *   **Train (60%)**: Model parameter learning.
    *   **Validation (20%)**: Hyperparameter tuning.
    *   **Test (20%)**: Final unbiased evaluation.
2.  **Hyperparameter Tuning**:
    *   **ALS**: Tuned `factors` (32, 64) and `regularization`. **Best**: `factors=32`, `reg=0.05`.
    *   **SVD**: Tuned `n_factors` (50, 100) and `lr_all`. **Best**: `n_factors=50`, `lr=0.01`.

### Current Results

*   **Validation Performance**:
    *   ALS Precision@10: **~9.7% - 10.4%**
*   **Test Set Performance (Unseen Data)**:
    *   **Precision@10: 8.64%**
    *   _Interpretation_: On average, nearly 1 in 10 recommended movies is a "hit" (rated 4+ stars) for the user, a solid baseline for offline evaluation.

## 6. Usage

### automated Training
To retrain the models on the full dataset using the optimal hyperparameters:

```bash
python src/train.py
```

This will generate the following artifacts in `models/`:
*   `als_artifacts.pkl`: ALS model and User-Item matrix.
*   `svd_model.pkl`: SVD ranking model.
*   `content_artifacts.pkl`: TF-IDF matrix and vectorizer.

### Prototyping
Explore `notebooks/02_Model_Training.ipynb` to see the step-by-step hyperparameter tuning and evaluation logic.

## 7. Next Steps (In Progress)
*   [ ] Implementation of `src/recommend.py` for the inference class.
*   [ ] Creation of FastAPI service (`src/main.py`).
*   [ ] Containerization (Docker).
