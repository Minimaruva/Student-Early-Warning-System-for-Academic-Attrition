from contextlib import asynccontextmanager
import pickle
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Define global storage for artifacts
ml_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load XGBoost Model
    # Note: xgb.Booster() is used for models saved as JSON
    try:
        model = xgb.Booster()
        model.load_model("../models/dropout_model_final.json")
        ml_artifacts["model"] = model
        print("INFO: Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        # In production, you might want to raise an error here to stop deployment

    # 2. Load Feature List
    try:
        with open("../models/model_features.pkl", "rb") as f:
            ml_artifacts["features"] = pickle.load(f)
        print("INFO: Features loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load features: {e}")

    yield
    
    # Cleanup on shutdown
    ml_artifacts.clear()

app = FastAPI(lifespan=lifespan)

# 3. Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. Restrict this in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Health Check Endpoint
@app.get("/health")
async def health():
    return {
        "status": "active",
        "model_loaded": "model" in ml_artifacts,
        "features_loaded": "features" in ml_artifacts
    }

# 5. Include Routers
# Assuming you have a file routers/predictions.py with a generic APIRouter
# from routers import predictions
# app.include_router(predictions.router)