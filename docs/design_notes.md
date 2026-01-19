# Design Notes

## Data & Features

### 1. Data Splitting & Validation
**Decision**: We implement a **Train / Validation / Test** split (60% / 20% / 20%) for the experimentation phase.

**Context**: 
- **Train (60%)**: Used to learn model parameters.
- **Validation (20%)**: Used for **Hyperparameter Optimization** (tuning factors, regularization, learning rates).
- **Test (20%)**: Used for final evaluation of the optimized models.
- **Production**: After selecting best parameters, we retrain on 100% of the data.

**Context**: 
- The chosen algorithms (Matrix Factorization via SVD and ALS) primarily rely on user-item interactions (ratings or confidence) and do not natively incorporate temporal dynamics in their standard implementations.
- The goal is to build a functional end-to-end MVP recommendations pipeline.

**Future Improvements (Post-MVP)**:
- **Temporal Split**: Use timestamps to split training and test data chronologically (e.g., train on first 80% of time, test on last 20%) to better simulate real-world performance.
- **Time Decay**: Implement a weighting scheme where recent ratings have higher influence on the model than older ones.
- **Sequential Models**: Explore algorithms that model user sessions or sequence history (e.g., RNNs/LSTMs or specialized sequential recommenders).

### 2. Handling Movie Genres
**Decision**: We will use **TF-IDF (Term Frequency-Inverse Document Frequency)** to create a content-based vector representation of movie genres.

**Context**:
- **Why TF-IDF?**: A simple multi-hot encoding treats all genres equally. TF-IDF weights rare genres (e.g., "Film-Noir", "Documentary") higher than common genres (e.g., "Drama", "Comedy"), providing a richer signal for similarity.
- **Implementation**: We treat the pipe-separated genre string (e.g., `"Action|Sci-Fi"`) as a "document".
- **Usage**: This is a critical component for our **Hybrid Candidate Generation** strategy:
    1.  **Candidate Expansion**: We will not rely solely on Collaborative Filtering (ALS). We will also fetch movies similar to the user's high-rated items using TF-IDF.
    2.  **Information Preservation**: This ensures we don't lose signal from explicit genre preferences, allowing the system to recommend relevant items even if the collaborative signal is weak.

**Reference**: See `tech_note_fe_genres.MD` for the detailed technical evaluation of Multi-Hot vs TF-IDF.
