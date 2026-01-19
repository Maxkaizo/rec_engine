# Technical Design Document: MovieRec Capstone Project

**Author**: Maximiliano Contieri
**Course**: Machine Learning Zoomcamp 2024
**Date**: 2026-01-16

---

## 1. Executive Summary

This document details the technical design of **MovieRec**, a movie recommendation system developed as the final project for the ML Zoomcamp. The project implements an industry-standard **two-stage pipeline (candidate generation and ranking)**, utilizing the **MovieLens 1M** dataset.

The objective is to demonstrate the ability to build a robust, efficient, and well-designed recommendation system that goes beyond the simple application of a single algorithm and is ready to be deployed as a real-world service.

---

## 2. System Architecture

The system's core is a two-stage pipeline that balances efficiency and accuracy.

```mermaid
graph TD
    A[User requests recommendations for user_id: 123] --> B{Stage 1: Candidate Generation};
    B --> C[Full Catalog (~3,700 movies)];
    C --> B;
    B -- ALS Model (implicit) --> D(List of ~150 Relevant Candidates);
    D --> E{Stage 2: Ranking};
    E -- SVD Model (surprise) --> F(Top 10 Accurately Ranked Movies);
    F --> G[Response to User];
```

### 2.1. Stage 1: Candidate Generation

- **Objective**: To efficiently narrow down the entire catalog of ~3,700 movies to a short, relevant list of ~150 candidates for a specific user.
- **Algorithm**: **Alternating Least Squares (ALS)** from the `implicit` library.
- **Justification**: ALS is extremely fast and optimized for implicit feedback, making it ideal for a first pass over the entire catalog. Its speed is crucial for maintaining low API latency.

### 2.2. Stage 2: Ranking

- **Objective**: To take the short list of 150 candidates and rank them as accurately as possible to generate the final Top-10 list.
- **Algorithm**: **Singular Value Decomposition (SVD)** from the `surprise` library.
- **Justification**: SVD excels at predicting the explicit rating (1-5 stars) a user would give a movie. By applying it only to the reduced set of candidates, we can afford to use a more accurate model without sacrificing overall performance.

### 2.3. Cold-Start Strategy

- **Problem**: What to recommend to a new user who is not in the training set?
- **Solution**: A separate endpoint will be implemented to return a list of the most popular movies in the catalog, based on a score that combines the average rating and the number of ratings. This ensures that new users receive sensible recommendations instead of an error.

---

## 3. Dataset and Preprocessing

- **Dataset**: **MovieLens 1M**.
- **Files Used**:
  - `ratings.dat`: Contains `UserID`, `MovieID`, `Rating`, `Timestamp`.
  - `movies.dat`: Contains `MovieID`, `Title`, `Genres`.
- **File Ignored**: `users.dat`.
  - **Justification**: As discussed, demographic data (age, gender, occupation) are weak and noisy predictors of movie tastes. **Collaborative filtering**, which relies purely on rating behavior, is much more powerful. Ignoring this file simplifies the model and focuses on the strongest signal: user-item interactions.

### 3.1. Preprocessing for ALS (Candidate Generation)

- The explicit ratings matrix will be transformed into an **implicit feedback** matrix. An interaction will be considered positive (confidence = 1) if `rating >= 4`, and negative (confidence = 0) otherwise. This adapts the data to the strengths of the `implicit` library.

### 3.2. Preprocessing for SVD (Ranking)

- The original explicit ratings (1-5) will be used, as `surprise` and SVD are designed for this type of data.

---

## 4. Model Evaluation (Offline)

Evaluation will be performed to validate the model's performance before deployment.

- **Split Strategy**: **Random Split (80% train, 20% test)**. The `train_test_split` function from `surprise` will be used to ensure a consistent and reproducible split.
- **Justification**: For a time-constrained project, a random split is the quickest and simplest way to get a robust evaluation without the complexity of a time-based split.

### 4.1. Evaluation Metrics

Two categories of metrics will be reported for a comprehensive evaluation:

1.  **Rating Prediction Metrics** (to evaluate the accuracy of the SVD ranker):
    - **RMSE (Root Mean Squared Error)**
    - **MAE (Mean Absolute Error)**

2.  **Ranking Metrics** (to evaluate the quality of the final Top-10 list):
    - **Precision@10**
    - **Recall@10**
    - **NDCG@10**

### 4.2. Baseline

All models will be compared against a simple **popularity** baseline to quantify the improvement achieved with the ML models.

---

## 5. API Design and Deployment

The system will be exposed via a RESTful API.

- **Framework**: **FastAPI** for its high performance and ease of use.
- **Containerization**: **Docker** to package the application and its dependencies, ensuring a consistent deployment.

### 5.1. Inference Flow

A key design point is inference efficiency:

1.  **API Startup**: The trained models (ALS and SVD) and the interaction matrix are loaded into memory **only once**.
2.  **Request**: The client sends a request with **only the `user_id`**.
3.  **Process**: The API uses the pre-loaded models to execute the two-stage pipeline in milliseconds, without needing to retrain or receive large volumes of data.

### 5.2. API Endpoints

- `POST /recommend`
  - **Body**: `{ "user_id": int }`
  - **Response**: A Top-10 list of recommended movies for that user, including `movie_id`, `title`, and `predicted_rating`.

- `GET /popular`
  - **Body**: None.
  - **Response**: A Top-10 list of the most popular movies (cold-start strategy).

- `GET /health`
  - **Body**: None.
  - **Response**: `200 OK` status with service health information.

---

## 6. Project Structure

```
/movie-rec-capstone
|-- data/
|   |-- ml-1m/
|   |   |-- ratings.dat
|   |   |-- movies.dat
|-- models/
|   |-- als_model.pkl
|   |-- svd_model.pkl
|-- notebooks/
|   |-- 01_EDA.ipynb
|   |-- 02_Model_Training.ipynb
|   |-- 03_Evaluation.ipynb
|-- src/
|   |-- main.py         # FastAPI app
|   |-- models.py       # Pydantic models
|   |-- recommend.py    # Core recommendation logic
|-- tests/
|   |-- test_api.py
|-- Dockerfile
|-- requirements.txt
|-- README.md
```
