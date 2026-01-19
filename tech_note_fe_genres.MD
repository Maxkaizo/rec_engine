# Technical Note: Feature Engineering for Movie Genres

## 1. Overview

This document outlines the technical strategy for converting the multi-label movie genres from the MovieLens dataset into a numerical format suitable for machine learning. The primary challenge is to represent genre combinations (e.g., `"Action|Adventure|Sci-Fi"`) in a way that preserves the relationships between movies that share some, but not all, genres.

## 2. The Challenge: Representing Multi-Label Data

A naive approach might treat each unique combination of genres as a single, distinct category. However, this method is flawed as it fails to recognize shared characteristics. For instance, it would treat `"Action|Comedy"` and `"Action|Drama"` as completely unrelated, despite both containing the "Action" genre. The goal is to create a vector representation where the similarity between vectors reflects the similarity in their underlying genres.

## 3. Evaluated Strategies

Two primary strategies were considered for this task: **Multi-Hot Encoding** and **TF-IDF (Term Frequency-Inverse Document Frequency)**.

### 3.1. Strategy 1: Multi-Hot Encoding

Multi-Hot Encoding creates a binary vector where each position corresponds to a unique genre. A `1` indicates the presence of a genre, and a `0` indicates its absence.

-   **Example**:
    -   `Action|Comedy` -> `[1, 1, 0, 0, ...]`
    -   `Action|Drama` -> `[1, 0, 1, 0, ...]`

| Pros | Cons |
| :--- | :--- |
| Simple and intuitive to implement. | **Treats all genres equally.** |
| Provides a direct, lossless representation. | Fails to capture the relative importance of genres. |

**The key drawback is its equal weighting.** A shared common genre like "Drama" contributes just as much to the similarity score as a shared niche genre like "Film-Noir." This dilutes the signal from more specific genres, which are often stronger indicators of user preference.

### 3.2. Strategy 2: TF-IDF (Recommended)

TF-IDF is a statistical measure from Natural Language Processing (NLP) that evaluates how relevant a word is to a document in a collection of documents. We can adapt this concept by treating each movie's genre string as a "document" and the set of all unique genres as the "vocabulary."

-   **Term Frequency (TF)**: How often a genre appears in a movie's genre list. In our case, this is always 1.
-   **Inverse Document Frequency (IDF)**: This is the crucial component. It provides a high weight to rare genres (like "Film-Noir") and a low weight to common genres (like "Drama").

**The result is a vector where the value for each genre is weighted by its specificity.**

| Pros | Cons |
| :--- | :--- |
| **Weights genres by importance.** | Slightly more complex than Multi-Hot Encoding. |
| Captures a more nuanced signal of similarity. |  |
| Aligns better with real-world recommendation logic. |  |

By using TF-IDF, the model learns that a shared niche genre is a much stronger signal of similarity than a shared common one, leading to more relevant and insightful content-based recommendations.

## 4. Final Decision and Implementation

For the content-based filtering component of this project, we will adopt the **TF-IDF strategy**. This decision is based on its ability to produce a more sophisticated and context-aware feature representation that prioritizes the most informative signals within the data.

### Implementation Snippet

The implementation will be handled using `scikit-learn`'s `TfidfVectorizer`.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Assume 'movies' is a pandas DataFrame with a 'genres' column

# 1. Pre-process the genre string for the vectorizer
movies["genres_str"] = movies["genres"].str.replace("|", " ", regex=False)

# 2. Initialize and fit the TF-IDF Vectorizer
# We use a custom token_pattern to ensure single-word genres are captured
tfidf = TfidfVectorizer(token_pattern=r"(?u)\b[A-Za-z-]+\b")
tfidf_matrix = tfidf.fit_transform(movies["genres_str"])

# 3. The resulting 'tfidf_matrix' is a sparse matrix where each row is a
# weighted vector representing a movie's genres. This matrix can now be
# used to compute cosine similarity for content-based recommendations.

# Example: Get similarity matrix
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

This approach provides a robust and effective foundation for the candidate generation stage of our recommendation pipeline.
