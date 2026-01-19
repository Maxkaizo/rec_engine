# Express Action Plan (3 Days): MovieRec Capstone

**Objective**: To complete and deliver a high-quality capstone project in 3 days, leveraging AI code assistants (GitHub Copilot, etc.) to accelerate development.

---

## Plan Philosophy

- **Parallelize Tasks**: Whenever possible, work on code and documentation simultaneously.
- **Rapid Iteration**: Don't aim for perfection on the first pass. It's better to have a functional end-to-end pipeline on day one and refine it later.
- **Delegate to AI**: Use AI assistants to generate boilerplate, write standard functions (e.g., evaluation), and speed up the creation of the API and Dockerfile.

---

## Day 1: Foundations and Modeling

**Day's Goal**: Have the data processed, baseline models trained and saved, and an initial understanding of their performance.

### Morning (3-4 hours): Setup and Exploratory Analysis

- **[ ] Task 1: Project Structure and Environment**
  - Create the folder structure (`data`, `models`, `notebooks`, `src`).
  - Set up the Python virtual environment (`venv` or `conda`).
  - Install the main libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
  - **AI Assistance**: Ask the AI to generate the initial `requirements.txt` file.

- **[ ] Task 2: Data Download and Loading**
  - Download the MovieLens 1M dataset and place it in the `data/ml-1m` folder.
  - Create the first notebook (`01_EDA.ipynb`).
  - Write the code to load `ratings.dat` and `movies.dat` into Pandas DataFrames.

- **[ ] Task 3: Exploratory Data Analysis (EDA)**
  - In the notebook, analyze the distributions of ratings, the number of ratings per user, and per movie.
  - Visualize the most common genres.
  - **AI Assistance**: Use prompts like "*Generate an exploratory data analysis for the MovieLens dataset*" or "*Create a plot of the rating distribution*."

### Afternoon (3-4 hours): Preprocessing and Training

- **[ ] Task 4: Data Preprocessing**
  - Create the second notebook (`02_Model_Training.ipynb`).
  - Prepare the data for `surprise`: Create a `Dataset` and a `trainset`.
  - Prepare the data for `implicit`: Create a user-item sparse matrix, converting ratings to implicit feedback (confidence).

- **[ ] Task 5: Model Training**
  - Install `surprise` and `implicit`.
  - Train the **SVD** model with `surprise` on the `trainset`.
  - Train the **ALS** model with `implicit` on the confidence matrix.
  - **AI Assistance**: Use the AI to get the boilerplate code for training an SVD model in Surprise and an ALS model in implicit.

### Evening (2 hours): Saving and Initial Predictions

- **[ ] Task 6: Save Trained Models**
  - Serialize and save the trained model objects (SVD and ALS) to the `models/` folder using `pickle`.

- **[ ] Task 7: Preliminary Inference Script**
  - Create a simple script in `src/` to load the models and test a prediction for a sample user. This validates that the models were saved and can be loaded correctly.

**Day's Deliverable**: Trained SVD and ALS models saved as `.pkl` files.

---

## Day 2: Pipeline, Evaluation, and API

**Day's Goal**: Have a complete recommendation pipeline, clear evaluation metrics, and a functional API in development.

### Morning (3-4 hours): Pipeline Implementation

- **[ ] Task 8: Two-Stage Pipeline Logic**
  - In a new file `src/recommend.py`, write the main function that:
    1.  Receives a `user_id`.
    2.  Loads the ALS and SVD models.
    3.  **Stage 1**: Uses the ALS model to generate 150 candidates.
    4.  **Stage 2**: Iterates over the candidates and uses the SVD model to predict a rating for each.
    5.  Ranks the candidates by the predicted rating and returns the Top-10.
  - **AI Assistance**: Describe the logic in a comment and ask the AI to generate the function.

### Afternoon (3-4 hours): Rigorous Evaluation

- **[ ] Task 9: Implementation of Evaluation Metrics**
  - Create the third notebook (`03_Evaluation.ipynb`).
  - Perform the 80/20 data split.
  - Calculate **RMSE** and **MAE** for the SVD model.
  - Write functions to calculate **Precision@10**, **Recall@10**, and **NDCG@10**.
  - **AI Assistance**: Ask the AI to generate the functions for the ranking metrics. Prompt: "*Write a Python function that calculates precision and recall @k for a list of predictions from the Surprise library*."

- **[ ] Task 10: Generate and Save Results**
  - Run the evaluation for the SVD model and a popularity baseline.
  - Save the results in a Markdown table or a JSON file.

### Evening (2-3 hours): API Development

- **[ ] Task 11: Create the FastAPI Application**
  - Install `fastapi` and `uvicorn`.
  - Create the `src/main.py` file.
  - Define the `POST /recommend` endpoint.
  - Integrate the logic from `src/recommend.py` into the endpoint.
  - Implement model loading on application startup (`@app.on_event("startup")`).
  - **AI Assistance**: Ask it to generate the basic structure of a FastAPI API with a POST endpoint.

**Day's Deliverable**: A functional evaluation script and a FastAPI API that can generate recommendations (though not yet in Docker).

---

## Day 3: Finalization, Deployment, and Documentation

**Day's Goal**: Have a complete, packaged, documented project ready for submission.

### Morning (3-4 hours): API Finalization and Docker

- **[ ] Task 12: Cold-Start Logic**
  - Implement the `GET /popular` endpoint that returns the most popular movies as a strategy for new users.

- **[ ] Task 13: Create the Dockerfile**
  - Create a `Dockerfile` for the application.
  - Ensure you copy the necessary files (`src/`, `models/`, `data/`).
  - Install dependencies from `requirements.txt`.
  - Define the `CMD` to start `uvicorn`.
  - **AI Assistance**: Ask it to generate a `Dockerfile` for a FastAPI application.

- **[ ] Task 14: Build and Test the Docker Image**
  - Build the Docker image locally (`docker build`).
  - Run the container (`docker run`).
  - Test the API running inside the container using `curl` or Postman.

### Afternoon (3-4 hours): Documentation

- **[ ] Task 15: Write the `README.md`**
  - This is the most important document for submission.
  - **Sections**: Introduction, Architecture, How to Run the Project (with Docker), Evaluation Results (include the metrics table), API Design.
  - **AI Assistance**: Use the TDD we already created as a base and ask the AI to summarize and adapt it into a README format.

- **[ ] Task 16: Clean Up Notebooks**
  - Ensure the notebooks are clean, with cells executed and outputs clear.
  - Add comments and Markdown text to explain each step.

### Evening (2 hours): Final Review and Submission

- **[ ] Task 17: Final Review**
  - Review all project files: code, notebooks, Dockerfile, README.
  - Ensure there are no hardcoded keys or sensitive information.
  - Check that the README instructions are clear and reproducible.

- **[ ] Task 18: Push to GitHub**
  - Create a repository on GitHub.
  - Push the entire project.
  - Verify that the README renders correctly.

**Day's Deliverable**: A complete GitHub repository with functional code, a Dockerfile, clean notebooks, and a professional `README.md`.
