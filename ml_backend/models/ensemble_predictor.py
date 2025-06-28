"""
Advanced Ensemble Predictor for IPL Win Probability
Combines multiple ML algorithms for robust predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

class EnsembleIPLPredictor:
    """Advanced ensemble predictor combining multiple ML algorithms"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.is_trained = False
        self.feature_names = []
        self.scaler = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all ML models"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=400,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=400,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'catboost': cb.CatBoostRegressor(
                iterations=300,
                depth=8,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        }
        
        logger.info(f"Initialized {len(self.models)} ML models")
    
    async def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame, y_val: pd.Series):
        """Train all models in the ensemble"""
        logger.info("Training ensemble models...")
        
        self.feature_names = list(X_train.columns)
        val_scores = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Validate
                val_pred = model.predict(X_val)
                val_score = r2_score(y_val, val_pred)
                val_scores[name] = val_score
                
                logger.info(f"{name} validation R2: {val_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                val_scores[name] = 0.0
        
        # Calculate ensemble weights based on validation performance
        self._calculate_weights(val_scores)
        self.is_trained = True
        
        logger.info("Ensemble training completed!")
    
    def _calculate_weights(self, val_scores: Dict[str, float]):
        """Calculate ensemble weights based on validation scores"""
        # Softmax weighting
        scores = np.array(list(val_scores.values()))
        scores = np.maximum(scores, 0.1)  # Minimum weight
        exp_scores = np.exp(scores * 3)  # Scale for more pronounced differences
        weights = exp_scores / np.sum(exp_scores)
        
        self.weights = dict(zip(val_scores.keys(), weights))
        
        logger.info(f"Model weights: {self.weights}")
    
    async def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Make ensemble prediction"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        # Convert features to DataFrame
        X = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0.0
        
        X = X[self.feature_names]  # Reorder columns
        
        # Get predictions from all models
        predictions = []
        for name, model in self.models.items():
            try:
                pred = model.predict(X)[0]
                predictions.append(pred * self.weights.get(name, 0.0))
            except Exception as e:
                logger.warning(f"Prediction error for {name}: {e}")
                predictions.append(0.0)
        
        # Ensemble prediction
        ensemble_pred = np.sum(predictions)
        ensemble_pred = np.clip(ensemble_pred, 0.01, 0.99)  # Clip to valid probability range
        
        return {
            'team1_win_prob': ensemble_pred,
            'team2_win_prob': 1.0 - ensemble_pred
        }
    
    async def get_individual_predictions(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get predictions from individual models"""
        X = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0.0
        
        X = X[self.feature_names]
        
        individual_preds = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X)[0]
                individual_preds[name] = float(np.clip(pred, 0.01, 0.99))
            except Exception as e:
                logger.warning(f"Individual prediction error for {name}: {e}")
                individual_preds[name] = 0.5
        
        return individual_preds
    
    def calculate_confidence(self, predictions: Dict[str, float]) -> str:
        """Calculate prediction confidence based on model agreement"""
        # Get individual predictions
        individual_preds = list(predictions.values())
        
        if len(individual_preds) < 2:
            return "low"
        
        # Calculate standard deviation of predictions
        std_dev = np.std(individual_preds)
        
        if std_dev < 0.05:
            return "high"
        elif std_dev < 0.15:
            return "medium"
        else:
            return "low"
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get average feature importance across models"""
        if not self.is_trained:
            return {}
        
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                weight = self.weights.get(name, 0.0)
                
                for i, feature in enumerate(self.feature_names):
                    if feature not in importance_dict:
                        importance_dict[feature] = 0.0
                    importance_dict[feature] += importances[i] * weight
        
        # Normalize
        total_importance = sum(importance_dict.values())
        if total_importance > 0:
            importance_dict = {k: v/total_importance for k, v in importance_dict.items()}
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def get_comprehensive_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for each model"""
        comprehensive_importance = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                model_importance = {}
                
                for i, feature in enumerate(self.feature_names):
                    model_importance[feature] = float(importances[i])
                
                comprehensive_importance[name] = model_importance
        
        return comprehensive_importance
    
    def save_models(self, filepath: str):
        """Save all trained models"""
        model_data = {
            'models': self.models,
            'weights': self.weights,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.weights = model_data['weights']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Models loaded from {filepath}")