"""
Advanced Ensemble Models for IPL Prediction
Implements Random Forest, XGBoost, LightGBM, CatBoost with hyperparameter tuning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import StandardScaler
import optuna
from datetime import datetime

logger = logging.getLogger(__name__)

class AdvancedEnsemblePredictor:
    """Advanced ensemble with hyperparameter optimization"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.weights = {}
        self.is_trained = False
        self.feature_names = []
        self.hyperparameters = {}
        
    def _get_model_configs(self) -> Dict[str, Dict]:
        """Get optimized model configurations"""
        return {
            'random_forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'xgboost': {
                'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 1],
                    'reg_lambda': [1, 1.5, 2]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 1],
                    'reg_lambda': [1, 1.5, 2]
                }
            },
            'catboost': {
                'model': cb.CatBoostRegressor(random_state=42, verbose=False),
                'params': {
                    'iterations': [200, 300, 500],
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'l2_leaf_reg': [1, 3, 5],
                    'border_count': [32, 64, 128]
                }
            }
        }
    
    async def train_with_hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                                             X_val: pd.DataFrame, y_val: pd.Series,
                                             use_optuna: bool = True):
        """Train models with advanced hyperparameter tuning"""
        logger.info("🚀 Starting advanced ensemble training with hyperparameter optimization...")
        
        self.feature_names = list(X_train.columns)
        model_configs = self._get_model_configs()
        val_scores = {}
        
        for name, config in model_configs.items():
            logger.info(f"🔧 Training {name} with hyperparameter optimization...")
            
            try:
                if use_optuna:
                    # Use Optuna for more advanced optimization
                    best_model = await self._optimize_with_optuna(
                        name, config, X_train, y_train, X_val, y_val
                    )
                else:
                    # Use GridSearchCV
                    best_model = await self._optimize_with_gridsearch(
                        config, X_train, y_train
                    )
                
                # Validate model
                val_pred = best_model.predict(X_val)
                val_score = r2_score(y_val, val_pred)
                val_scores[name] = val_score
                
                # Store model
                self.models[name] = best_model
                
                logger.info(f"✅ {name} - R² Score: {val_score:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error training {name}: {e}")
                val_scores[name] = 0.0
        
        # Calculate ensemble weights
        self._calculate_advanced_weights(val_scores)
        self.is_trained = True
        
        # Log final ensemble performance
        ensemble_pred = await self._ensemble_predict(X_val)
        ensemble_score = r2_score(y_val, ensemble_pred)
        logger.info(f"🎯 Final Ensemble R² Score: {ensemble_score:.4f}")
        
        return val_scores
    
    async def _optimize_with_optuna(self, model_name: str, config: Dict,
                                  X_train: pd.DataFrame, y_train: pd.Series,
                                  X_val: pd.DataFrame, y_val: pd.Series):
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
                model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
                
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2)
                }
                model = xgb.XGBRegressor(random_state=42, n_jobs=-1, **params)
                
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2)
                }
                model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **params)
                
            elif model_name == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 500),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10)
                }
                model = cb.CatBoostRegressor(random_state=42, verbose=False, **params)
            
            # Train and evaluate
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            return r2_score(y_val, pred)
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, timeout=300)  # 5 minutes max
        
        # Train best model
        best_params = study.best_params
        self.hyperparameters[model_name] = best_params
        
        if model_name == 'random_forest':
            best_model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
        elif model_name == 'xgboost':
            best_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, **best_params)
        elif model_name == 'lightgbm':
            best_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **best_params)
        elif model_name == 'catboost':
            best_model = cb.CatBoostRegressor(random_state=42, verbose=False, **best_params)
        
        best_model.fit(X_train, y_train)
        return best_model
    
    async def _optimize_with_gridsearch(self, config: Dict, X_train: pd.DataFrame, y_train: pd.Series):
        """Optimize hyperparameters using GridSearchCV"""
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=3,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    def _calculate_advanced_weights(self, val_scores: Dict[str, float]):
        """Calculate sophisticated ensemble weights"""
        # Remove models with poor performance
        filtered_scores = {k: v for k, v in val_scores.items() if v > 0.1}
        
        if not filtered_scores:
            # Fallback to equal weights
            self.weights = {k: 1/len(val_scores) for k in val_scores.keys()}
            return
        
        # Softmax with temperature scaling
        scores = np.array(list(filtered_scores.values()))
        temperature = 2.0  # Controls weight distribution
        exp_scores = np.exp(scores * temperature)
        weights = exp_scores / np.sum(exp_scores)
        
        # Assign weights
        self.weights = {}
        for i, model_name in enumerate(filtered_scores.keys()):
            self.weights[model_name] = weights[i]
        
        # Zero weight for poor models
        for model_name in val_scores.keys():
            if model_name not in filtered_scores:
                self.weights[model_name] = 0.0
        
        logger.info(f"🎯 Ensemble weights: {self.weights}")
    
    async def _ensemble_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble prediction"""
        predictions = []
        total_weight = 0
        
        for name, model in self.models.items():
            if name in self.weights and self.weights[name] > 0:
                pred = model.predict(X)
                weight = self.weights[name]
                predictions.append(pred * weight)
                total_weight += weight
        
        if total_weight == 0:
            # Fallback to first available model
            first_model = list(self.models.values())[0]
            return first_model.predict(X)
        
        ensemble_pred = np.sum(predictions, axis=0) / total_weight
        return ensemble_pred
    
    async def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Make ensemble prediction"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0.0
        
        X = X[self.feature_names]
        
        # Make ensemble prediction
        ensemble_pred = await self._ensemble_predict(X)
        team1_prob = float(np.clip(ensemble_pred[0], 0.01, 0.99))
        
        return {
            'team1_win_prob': team1_prob,
            'team2_win_prob': 1.0 - team1_prob
        }
    
    async def get_individual_predictions(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get predictions from individual models"""
        X = pd.DataFrame([features])
        
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
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get ensemble feature importance"""
        if not self.is_trained:
            return {}
        
        importance_dict = {}
        total_weight = sum(self.weights.values())
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_') and name in self.weights:
                importances = model.feature_importances_
                weight = self.weights[name] / total_weight if total_weight > 0 else 0
                
                for i, feature in enumerate(self.feature_names):
                    if feature not in importance_dict:
                        importance_dict[feature] = 0.0
                    importance_dict[feature] += importances[i] * weight
        
        # Normalize
        total_importance = sum(importance_dict.values())
        if total_importance > 0:
            importance_dict = {k: v/total_importance for k, v in importance_dict.items()}
        
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save_models(self, filepath: str):
        """Save ensemble models"""
        model_data = {
            'models': self.models,
            'weights': self.weights,
            'feature_names': self.feature_names,
            'hyperparameters': self.hyperparameters,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"💾 Ensemble models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load ensemble models"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.weights = model_data['weights']
        self.feature_names = model_data['feature_names']
        self.hyperparameters = model_data.get('hyperparameters', {})
        self.is_trained = model_data['is_trained']
        
        logger.info(f"📂 Ensemble models loaded from {filepath}")