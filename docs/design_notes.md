# Design Notes

## Data & Features

### 1. Dropping `Timestamp` from Ratings
**Decision**: We are excluding the `Timestamp` column from the input data for both the SVD and ALS models in the initial MVP.

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
- **Usage**: This allows us to find "Similar Items" based purely on content, which is crucial for:
    1.  **Cold Start**: Recommending similar movies to a user's initial selection before we have collaborative history.
    2.  **Hybridization**: Potential to mix content-based scores with ALS/SVD scores.

**Reference**: See `tech_note_fe_genres.MD` for the detailed technical evaluation of Multi-Hot vs TF-IDF.
