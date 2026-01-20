
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
├── deployment/             # Kubernetes manifests (Deployment, Service, HPA)
├── docs/                   # Documentation and guides
├── models/                 # Saved model artifacts (.pkl)
├── notebooks/              # Jupyter notebooks (EDA and Training)
├── src/                    # Source code
│   ├── main.py             # FastAPI service
│   ├── recommend.py        # Recommender system logic
│   ├── train.py            # Model training script
│   ├── test_api.py         # API integration test
│   └── test_recommend.py   # Modular inference test
├── Dockerfile              # Container definition
├── pyproject.toml          # Project dependencies
└── README.md               # Main documentation
```

## 3. Setup and Installation

This project uses **uv** for fast and reliable dependency management.

### Prerequisites
*   Linux/macOS (or WSL on Windows)
*   Python 3.12+
*   `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
*   **For Kubernetes (Optional)**:
    *   `kind`: [Installation Guide](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
    *   `kubectl`: [Installation Guide](https://kubernetes.io/docs/tasks/tools/)

### Installation Steps
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Maxkaizo/rec_engine.git
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
The **MovieLens 1M** dataset is already included in the repository under `data/ml-1m/`. No additional download is required to run the training or inference scripts.

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

## 6. Model Usage (Training)

To retrain the models on the full dataset using the optimal hyperparameters:

```bash
python src/train.py
```

## 7. Deployment

### Docker
The application is fully containerized.
1.  **Build**: `docker build -t rec-engine .`
2.  **Run**: `docker run -d -p 8000:8000 rec-engine`

### Kubernetes
For container orchestration, high availability, and auto-scaling, see the [Kubernetes Deployment Guide](docs/kubernetes_guide.md).

**Quick Automated kubernetes' Setup**:
```bash
./scripts/setup_k8s.sh
```

## 8. Verification and Testing

We provide multiple ways to verify the system depending on how you've deployed it.

### 1. Logic Verification (No Server)
To test the core recommendation engine and pipeline logic without starting a web server:
```bash
python src/test_recommend.py
```

### 2. API Verification (Integration)
The API must be running for these tests. Choose the scenario that matches your deployment:

| Scenario | Start API Command | Test Command |
| :--- | :--- | :--- |
| **Local Python** | `python src/main.py` | `python src/test_api.py --platform docker` |
| **Docker** | `docker run -d -p 8000:8000 rec-engine` | `python src/test_api.py --platform docker` |
| **Kubernetes** | `./scripts/setup_k8s.sh` | `python src/test_api.py --platform k8s` |

---
**Note for Kubernetes**: When testing on K8s, the setup script automates the deployment. However, you must ensure the **port-forward** command is running in a separate terminal before executing the test script:
```bash
kubectl port-forward service/rec-engine-service 8080:80
```

## 9. Final Review Checklist (Rubric Compliance)
*   [x] **Problem Description**: Detailed in Section 1.
*   [x] **EDA**: Extensive analysis in `01_EDA.ipynb`.
*   [x] **Model Training**: Multiple models (ALS, SVD) tuned in `02_Model_Training.ipynb`.
*   [x] **Export to Script**: Logic formalized in `src/train.py`.
*   [x] **Reproducibility**: `uv` environment and clear data instructions provided.
*   [x] **Model Deployment**: FastAPI service in `src/main.py`.
*   [x] **Dependency Management**: `pyproject.toml` and `uv.lock`.
*   [x] **Containerization**: `Dockerfile` provided and documented.
*   [x] **Cloud/Kubernetes Deployment**: Manifiestos (Deployment, Service, HPA) and automation script `scripts/setup_k8s.sh` provided and verified.
