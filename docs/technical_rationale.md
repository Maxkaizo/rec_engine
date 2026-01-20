# Technical Rationale: Algorithms and Libraries

This document explains the choice of algorithms and libraries used in this project, providing justification for their selection as they extend beyond the standard scope of the DataTalks.Club generic Machine Learning Bootcamp.

## 1. Collaborative Filtering Algorithms

### ALS (Alternating Least Squares)
*   **Source Library**: `implicit`
*   **Why we used it**: ALS is the industry standard for **Implicit Feedback**. While MovieLens provides explicit ratings, treating interactions as implicit signals allows for extremely fast candidate retrieval.
*   **How it Works**: ALS is a **Matrix Factorization** technique. It breaks down the large, sparse User-Item interaction matrix into two smaller, dense matrices: one for Users and one for Items, each representing "latent factors" (hidden characteristics). 
    *   It "alternates" between fixing the User matrix to optimize the Item matrix, and then fixing the Item matrix to optimize the User matrix using Least Squares. 
    *   The "dot product" of a user vector and an item vector represents the predicted preference.

### SVD (Singular Value Decomposition)
*   **Source Library**: `scikit-surprise`
*   **Why we used it**: SVD is the state-of-the-art for **Explicit Rating Prediction** (predicting 1-5 stars).
*   **How it Works**: Similar to ALS, it uses **Matrix Factorization**, but specifically optimized for sparse explicit data (minimizing RMSE). 
    *   It learns a "bias" for each user (some people are harsh critics) and each item (some movies are universally loved), plus the interaction between their latent factors.
    *   Formula: $\hat{r}_{ui} = \mu + b_u + b_i + q_i^T p_u$
    *   It uses **Stochastic Gradient Descent (SGD)** to minimize the error between predicted and actual ratings during training.

## 2. Content-Based Filtering

### TF-IDF on Movie Genres
*   **Source Library**: `scikit-learn`
*   **Why we used it**: To maintain topical relevance and handle users with very specific tastes that collaborative filtering might initially miss.
*   **How it Works**: 
    1.  **Term Frequency (TF)**: Counts how often a genre (e.g., "Sci-Fi") appears in a movie's metadata.
    2.  **Inverse Document Frequency (IDF)**: Penalizes very common genres (like "Drama") and rewards unique ones (like "Film-Noir") to make the similarity more meaningful.
    3.  **Cosine Similarity**: We calculate the "distance" between movie vectors in this TF-IDF space. If a user likes a movie with a specific genre vector, we recommend other movies with the most similar vectors.

## 3. Cold Start Strategy: Popularity-Based Fallback

When a new user visits the platform, the recommender system has no historical data to compute latent factors (ALS/SVD) or content preferences (TF-IDF). This is the **Cold Start Problem**.

*   **Why Popularity?**: In the absence of user-specific data, "Popularity with a quality threshold" is the most robust baseline. It relies on the **Wisdom of the Crowds**: if a large number of diverse users highly rate a movie, it has a high probability of being enjoyed by a new user.
*   **How it Works**: We don't just use the most-rated movies (which can be biased by marketing), but a **weighted popularity** (Top 50 movies with >100 ratings, sorted by average score). This ensures the fallback recommendations are both widely recognized and high-quality.

## 4. Libraries Justification

### `implicit`
Contrary to standard `scikit-learn`, `implicit` is written in Cython and C++, making it significantly faster for training and recommendation generation. In a production-like scenario, serving time is critical, and `implicit` can generate hundreds of candidates in milliseconds.

### `scikit-surprise`
Standard `scikit-learn` does not have built-in support for the specific data structures (User-Item-Rating triplets) used in recommender systems. `scikit-surprise` is a specialized toolkit that provides:
1.  **optimized Data Loaders**: Efficient handling of sparse rating data.
2.  **Built-in SVD/SVD++**: Pre-optimized implementations of state-of-the-art algorithms.
3.  **Cross-Validation**: Specialized cross-validation for recommendation metrics like RMSE and MAE.

### `FastAPI`
While Flask is taught in the course, **FastAPI** was chosen for this capstone because:
*   **Asynchronous Support**: Better handling of concurrent requests.
*   **Auto-Documentation**: Generates Swagger/OpenAPI docs automatically (crucial for peer review).
*   **Type Safety**: Uses Pydantic to ensure the data coming into the recommender is valid.

### `uv`
We used `uv` as the package manager because it is significantly faster than `pip` and provides a more robust dependency resolution (`uv lock`), ensuring that anyone reviewing the project gets the exact same environment setup.

## 5. Architectural Choice: Two-Stage Pipeline
Combining ALS and SVD into a hybrid pipeline is a "production-grade" pattern. It balances the high-speed retrieval of ALS with the high-precision ranking of SVD, a pattern commonly used by platforms like YouTube and Pinterest to handle large datasets efficiently.
