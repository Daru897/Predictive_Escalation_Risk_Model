import joblib
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
from contextlib import asynccontextmanager

# ----------------------------
# Setup Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# Configuration - FIXED PATHS
# ----------------------------
# Get the absolute path to the models directory
MODELS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(MODELS_ROOT, "models", "artifacts")
EXPLAIN_DIR = os.path.join(ARTIFACTS_DIR, "explainability")

# Ensure directories exist
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(EXPLAIN_DIR, exist_ok=True)

MODEL_CACHE = {}
FEATURE_CACHE = {}

# ----------------------------
# Lifespan Management
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup: Load models and data
    startup_time = time.time()
    logger.info("🚀 Starting application startup...")
    logger.info(f"📁 Looking for models in: {ARTIFACTS_DIR}")
    
    try:
        # Check if artifacts directory exists and has files
        if not os.path.exists(ARTIFACTS_DIR):
            logger.error(f"❌ Artifacts directory does not exist: {ARTIFACTS_DIR}")
            raise RuntimeError(f"Artifacts directory not found: {ARTIFACTS_DIR}")
        
        # List files in artifacts directory for debugging
        artifact_files = os.listdir(ARTIFACTS_DIR)
        logger.info(f"📋 Files in artifacts directory: {artifact_files}")
        
        load_models()
        load_explanations()
        logger.info(f"✅ Startup completed in {time.time() - startup_time:.2f}s")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Don't raise here, allow the API to start without models for debugging
        logger.warning("⚠️ API starting without models for debugging")
    
    yield  # Application runs here
    
    # Shutdown: Cleanup
    logger.info("🛑 Shutting down application...")
    MODEL_CACHE.clear()
    FEATURE_CACHE.clear()

# ----------------------------
# Model Loading
# ----------------------------
def load_models():
    """Load all available models with caching"""
    model_paths = [
        ("xgboost", os.path.join(ARTIFACTS_DIR, "xgboost.joblib")),
        ("random_forest", os.path.join(ARTIFACTS_DIR, "random_forest.joblib")),
        ("logistic_regression", os.path.join(ARTIFACTS_DIR, "logistic_regression.joblib"))
    ]
    
    loaded_models = {}
    models_found = 0
    
    for model_name, path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                loaded_models[model_name] = {
                    "model": model,
                    "path": path,
                    "loaded_at": datetime.now().isoformat(),
                    "type": type(model).__name__
                }
                models_found += 1
                logger.info(f"✅ Loaded {model_name} from {path}")
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name}: {e}")
        else:
            logger.warning(f"⚠️ Model file not found: {path}")
    
    if not loaded_models:
        logger.warning("⚠️ No models loaded. API will run in limited mode.")
        # Don't raise error, allow API to start for debugging
        return {}
    
    MODEL_CACHE.update(loaded_models)
    return loaded_models

def load_explanations():
    """Load explanation data"""
    top_features_path = os.path.join(EXPLAIN_DIR, "ticket_top_features.json")
    if os.path.exists(top_features_path):
        try:
            with open(top_features_path, "r") as f:
                data = json.load(f)
                FEATURE_CACHE["ticket_top_features"] = data
                logger.info(f"✅ Loaded explanations for {len(data.get('ticket_features', []))} tickets")
        except Exception as e:
            logger.error(f"❌ Failed to load explanations: {e}")
    else:
        logger.warning(f"⚠️ No explanation data found at {top_features_path}")

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_features(features: Dict[str, Any], model_name: str) -> pd.DataFrame:
    """Preprocess input features to match model expectations"""
    try:
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Get expected feature order from model metadata if available
        metadata_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_metadata.json")
        expected_features = None
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    expected_features = metadata.get("feature_names")
            except Exception as e:
                logger.warning(f"⚠️ Could not load metadata: {e}")
        
        # If expected features are known, ensure proper order and missing features
        if expected_features:
            # Add missing features with default values
            for feature in expected_features:
                if feature not in X.columns:
                    X[feature] = 0  # Default value for missing features
            
            # Select only expected features in correct order
            X = X[expected_features]
        else:
            # Fallback: use the features provided
            logger.warning("⚠️ No feature metadata found, using provided features as-is")
        
        # Ensure numeric types
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = X.fillna(0)
        
        logger.info(f"🔧 Preprocessed features: {X.shape}")
        return X
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        raise ValueError(f"Feature preprocessing failed: {e}")

# ----------------------------
# Prediction Logic
# ----------------------------
def predict_escalation_risk(model, X: pd.DataFrame, model_type: str) -> float:
    """Make prediction with appropriate method based on model type"""
    try:
        if hasattr(model, "predict_proba"):
            # Use probability for models that support it
            proba = model.predict_proba(X)
            # Handle binary classification (return probability of class 1)
            if len(proba.shape) > 1 and proba.shape[1] > 1:
                return float(proba[0, 1])
            else:
                return float(proba[0])
        else:
            # Fallback to predict for models without predict_proba
            prediction = model.predict(X)
            return float(prediction[0])
            
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise ValueError(f"Prediction failed: {e}")

def get_explanation(ticket_id: int) -> List[Dict]:
    """Get explanation for a specific ticket"""
    if "ticket_top_features" not in FEATURE_CACHE:
        return []
    
    explanations = FEATURE_CACHE["ticket_top_features"].get("ticket_features", [])
    for ticket in explanations:
        if ticket.get("ticket_id") == ticket_id:
            return ticket.get("top_features", [])[:3]  # Return top 3 features
    
    return []

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(
    title="Escalation Risk Inference API",
    description="API for predicting ticket escalation risk with explainability",
    version="1.0.0",
    lifespan=lifespan
)

class TicketFeatures(BaseModel):
    ticket_id: int
    features: Dict[str, Any]
    model_name: Optional[str] = "xgboost"  # Allow model selection

class PredictionResponse(BaseModel):
    ticket_id: int
    risk_score: float
    model_used: str
    top_features: List[Dict]
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    available_models: List[str]
    loaded_at: str
    explanations_loaded: bool
    artifacts_directory: str

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if MODEL_CACHE else "degraded",
        "available_models": list(MODEL_CACHE.keys()),
        "loaded_at": datetime.now().isoformat(),
        "explanations_loaded": "ticket_top_features" in FEATURE_CACHE,
        "artifacts_directory": ARTIFACTS_DIR
    }

@app.get("/models")
async def list_models():
    """List available models"""
    models_info = []
    for name, info in MODEL_CACHE.items():
        models_info.append({
            "name": name,
            "type": info["type"],
            "loaded_at": info["loaded_at"],
            "path": info["path"]
        })
    return {"models": models_info}

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TicketFeatures):
    """Predict escalation risk for a ticket"""
    start_time = time.time()
    
    # Check if models are loaded
    if not MODEL_CACHE:
        raise HTTPException(
            status_code=503,
            detail="No models loaded. Please check the artifacts directory and restart the API."
        )
    
    try:
        # Validate model selection
        model_name = input_data.model_name.lower()
        if model_name not in MODEL_CACHE:
            available_models = list(MODEL_CACHE.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' not available. Choose from: {available_models}"
            )
        
        model_info = MODEL_CACHE[model_name]
        model = model_info["model"]
        
        # Preprocess features
        X = preprocess_features(input_data.features, model_name)
        
        # Make prediction
        risk_score = predict_escalation_risk(model, X, model_name)
        
        # Get explanation
        explanation = get_explanation(input_data.ticket_id)
        
        # Log prediction
        logger.info(f"📊 Prediction - Ticket: {input_data.ticket_id}, "
                   f"Model: {model_name}, Risk: {risk_score:.3f}, "
                   f"Time: {time.time() - start_time:.3f}s")
        
        return {
            "ticket_id": input_data.ticket_id,
            "risk_score": risk_score,
            "model_used": model_name,
            "top_features": explanation,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/explanations/{ticket_id}")
async def get_ticket_explanation(ticket_id: int):
    """Get explanation for a specific ticket"""
    explanation = get_explanation(ticket_id)
    if not explanation:
        raise HTTPException(
            status_code=404,
            detail=f"No explanation found for ticket {ticket_id}"
        )
    
    return {
        "ticket_id": ticket_id,
        "explanations": explanation,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/debug/paths")
async def debug_paths():
    """Debug endpoint to check file paths"""
    return {
        "models_root": MODELS_ROOT,
        "artifacts_dir": ARTIFACTS_DIR,
        "explain_dir": EXPLAIN_DIR,
        "artifacts_exists": os.path.exists(ARTIFACTS_DIR),
        "artifacts_files": os.listdir(ARTIFACTS_DIR) if os.path.exists(ARTIFACTS_DIR) else [],
        "current_directory": os.getcwd(),
        "script_directory": os.path.dirname(os.path.abspath(__file__))
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error": "HTTP Error"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": "Server Error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_keep_alive=300
    )