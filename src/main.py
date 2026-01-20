
from fastapi import FastAPI, HTTPException, Query
from src.recommend import RecommenderSystem
import uvicorn
import os

app = FastAPI(
    title="Movie Recommender API",
    description="Two-stage movie recommendation system using ALS Retrieval and SVD Ranking.",
    version="1.0.0"
)

# Global Recommender Instance
# Loaded on startup to serve requests faster
rec_sys = None

@app.on_event("startup")
def load_models():
    global rec_sys
    models_path = os.getenv("MODELS_DIR", "models")
    print(f"Loading recommender system from {models_path}...")
    try:
        rec_sys = RecommenderSystem(models_dir=models_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        # In production, we might want to shut down or retry
        raise RuntimeError("Failed to load recommendation models")

@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": rec_sys is not None}

@app.get("/recommend/{user_id}")
def get_recommendations(
    user_id: int, 
    k: int = Query(default=10, ge=1, le=50),
    enrich: bool = Query(default=False, description="Set to true to return movie titles and genres")
):
    """
    Get personalized movie recommendations for a user.
    - **user_id**: The ID of the user (e.g. 1, 10, 100).
    - **k**: Number of recommendations to return (max 50).
    - **enrich**: Whether to return full metadata or just raw IDs.
    """
    if rec_sys is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    try:
        recommendations = rec_sys.get_recommendations(user_id=user_id, k=k, enrich=enrich)
        return {
            "user_id": user_id,
            "count": len(recommendations),
            "enriched": enrich,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/popular")
def get_popular(k: int = Query(default=10, ge=1, le=50)):
    """
    Get top popular movies (fallback for cold start).
    """
    if rec_sys is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    try:
        # We use enrich=True for popular by default as it's a discovery endpoint
        popular = rec_sys.get_recommendations(user_id=-1, k=k, enrich=True)
        return {
            "type": "popular_fallback",
            "count": len(popular),
            "recommendations": popular
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
