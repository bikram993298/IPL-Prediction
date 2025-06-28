#!/usr/bin/env python3
"""
IPL Win Probability Prediction - Advanced ML Backend
Comprehensive machine learning system with multiple algorithms and real-time predictions
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.ensemble_predictor import EnsembleIPLPredictor
from models.deep_learning_models import DeepLearningIPLModel
from models.time_series_predictor import TimeSeriesPredictor
from data.data_processor import AdvancedDataProcessor
from data.feature_engineering import FeatureEngineer
from data.live_data_collector import LiveDataCollector
from utils.model_manager import ModelManager
from utils.performance_monitor import PerformanceMonitor
from api.cricket_data_api import CricketDataAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_backend.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="IPL ML Prediction API",
    description="Advanced Machine Learning API for IPL Win Probability Predictions",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ML components
ml_predictor = None
data_processor = None
feature_engineer = None
model_manager = None
performance_monitor = None
live_data_collector = None

# Pydantic models for API
class MatchState(BaseModel):
    team1: str
    team2: str
    current_score: int
    wickets: int
    overs: int
    balls: int
    target: Optional[int] = None
    venue: str
    weather: str
    toss_winner: str
    toss_decision: str
    is_first_innings: bool = True

class PredictionResponse(BaseModel):
    team1_probability: float
    team2_probability: float
    confidence: str
    factors: Dict[str, float]
    model_predictions: Dict[str, float]
    feature_importance: Dict[str, float]
    prediction_timestamp: str

class ModelTrainingRequest(BaseModel):
    model_type: str
    hyperparameters: Dict[str, Any]
    training_data_size: int = 10000

class LiveMatchData(BaseModel):
    match_id: str
    team1: str
    team2: str
    current_score: int
    wickets: int
    overs: int
    balls: int
    target: Optional[int]
    status: str
    venue: str

@app.on_event("startup")
async def startup_event():
    """Initialize ML components on startup"""
    global ml_predictor, data_processor, feature_engineer, model_manager, performance_monitor, live_data_collector
    
    logger.info("Initializing ML Backend...")
    
    try:
        # Initialize components
        data_processor = AdvancedDataProcessor()
        feature_engineer = FeatureEngineer()
        model_manager = ModelManager()
        performance_monitor = PerformanceMonitor()
        live_data_collector = LiveDataCollector()
        
        # Initialize ML predictor
        ml_predictor = EnsembleIPLPredictor()
        
        # Load or train models
        await model_manager.load_or_train_models()
        
        logger.info("ML Backend initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize ML Backend: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "IPL ML Prediction API",
        "version": "2.0.0",
        "status": "active",
        "models_loaded": len(model_manager.models) if model_manager else 0,
        "endpoints": {
            "predict": "/predict",
            "live_matches": "/live-matches",
            "train_model": "/train-model",
            "model_performance": "/model-performance",
            "feature_importance": "/feature-importance"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_win_probability(match_state: MatchState):
    """Predict win probability for given match state"""
    try:
        logger.info(f"Prediction request: {match_state.team1} vs {match_state.team2}")
        
        # Process match state
        processed_data = await data_processor.process_match_state(match_state.dict())
        
        # Engineer features
        features = await feature_engineer.create_features(processed_data)
        
        # Get predictions from ensemble
        predictions = await ml_predictor.predict(features)
        
        # Calculate confidence
        confidence = ml_predictor.calculate_confidence(predictions)
        
        # Get individual model predictions
        model_predictions = await ml_predictor.get_individual_predictions(features)
        
        # Get feature importance
        feature_importance = ml_predictor.get_feature_importance()
        
        # Calculate match factors
        factors = {
            "momentum": float(features.get('momentum_score', 5.0)),
            "pressure": float(features.get('pressure_index', 5.0)),
            "form": float(features.get('team_form_score', 5.0)),
            "conditions": float(features.get('conditions_score', 5.0))
        }
        
        response = PredictionResponse(
            team1_probability=float(predictions['team1_win_prob']),
            team2_probability=float(predictions['team2_win_prob']),
            confidence=confidence,
            factors=factors,
            model_predictions=model_predictions,
            feature_importance=feature_importance,
            prediction_timestamp=datetime.now().isoformat()
        )
        
        # Log prediction for monitoring
        await performance_monitor.log_prediction(match_state.dict(), response.dict())
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/live-matches")
async def get_live_matches():
    """Get current live IPL matches"""
    try:
        live_matches = await live_data_collector.get_live_matches()
        return {"matches": live_matches, "count": len(live_matches)}
    except Exception as e:
        logger.error(f"Live matches error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch live matches: {str(e)}")

@app.post("/train-model")
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """Train a new model with specified parameters"""
    try:
        # Add training task to background
        background_tasks.add_task(
            model_manager.train_new_model,
            request.model_type,
            request.hyperparameters,
            request.training_data_size
        )
        
        return {
            "message": f"Training {request.model_type} model started",
            "status": "training_initiated",
            "estimated_time": "10-30 minutes"
        }
    except Exception as e:
        logger.error(f"Model training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/model-performance")
async def get_model_performance():
    """Get performance metrics for all models"""
    try:
        performance_data = await performance_monitor.get_performance_summary()
        return performance_data
    except Exception as e:
        logger.error(f"Performance data error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance data: {str(e)}")

@app.get("/feature-importance")
async def get_feature_importance():
    """Get feature importance across all models"""
    try:
        importance_data = ml_predictor.get_comprehensive_feature_importance()
        return {"feature_importance": importance_data}
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(model_manager.models) if model_manager else 0,
        "memory_usage": performance_monitor.get_memory_usage() if performance_monitor else "unknown"
    }

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )