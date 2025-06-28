"""
Model Management System
Handles model training, saving, loading, and version control
"""

import os
import json
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from models.ensemble_predictor import EnsembleIPLPredictor
from models.deep_learning_models import DeepLearningIPLModel
from data.data_processor import AdvancedDataProcessor
from data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class ModelManager:
    """Comprehensive model management system"""
    
    def __init__(self, models_dir: str = "models/saved"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.model_metadata = {}
        self.data_processor = AdvancedDataProcessor()
        self.feature_engineer = FeatureEngineer()
        
    async def load_or_train_models(self):
        """Load existing models or train new ones"""
        logger.info("Loading or training models...")
        
        # Check for existing models
        model_files = list(self.models_dir.glob("*.pkl"))
        
        if model_files:
            await self._load_existing_models()
        else:
            await self._train_new_models()
    
    async def _load_existing_models(self):
        """Load existing trained models"""
        logger.info("Loading existing models...")
        
        for model_file in self.models_dir.glob("*.pkl"):
            try:
                model_name = model_file.stem
                model = joblib.load(model_file)
                self.models[model_name] = model
                
                # Load metadata if exists
                metadata_file = model_file.with_suffix('.json')
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        self.model_metadata[model_name] = json.load(f)
                
                logger.info(f"Loaded model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
    
    async def _train_new_models(self):
        """Train new models from scratch"""
        logger.info("Training new models...")
        
        # Generate training data
        training_data = await self.data_processor.generate_training_data(size=15000)
        
        # Process features
        X, y = await self._prepare_training_data(training_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Train ensemble model
        ensemble_model = EnsembleIPLPredictor()
        await ensemble_model.train(X_train, y_train, X_val, y_val)
        
        # Save ensemble model
        await self._save_model(ensemble_model, "ensemble_predictor", {
            "model_type": "ensemble",
            "training_size": len(X_train),
            "validation_score": "calculated_during_training",
            "features": list(X.columns)
        })
        
        # Train deep learning model
        dl_model = DeepLearningIPLModel(model_type='tensorflow')
        await dl_model.train(X_train, y_train, X_val, y_val, epochs=50)
        
        # Save deep learning model
        await self._save_model(dl_model, "deep_learning_tensorflow", {
            "model_type": "deep_learning",
            "framework": "tensorflow",
            "training_size": len(X_train),
            "epochs": 50,
            "features": list(X.columns)
        })
        
        self.models["ensemble_predictor"] = ensemble_model
        self.models["deep_learning_tensorflow"] = dl_model
        
        logger.info("Model training completed!")
    
    async def _prepare_training_data(self, training_data: pd.DataFrame):
        """Prepare training data with feature engineering"""
        logger.info("Preparing training data...")
        
        features_list = []
        targets = []
        
        for _, row in training_data.iterrows():
            # Convert row to dictionary
            match_data = row.to_dict()
            
            # Create features
            features = await self.feature_engineer.create_features(match_data)
            features_list.append(features)
            targets.append(row.get('win_probability', 0.5))
        
        # Convert to DataFrame
        X = pd.DataFrame(features_list)
        y = pd.Series(targets)
        
        # Fill missing values
        X = X.fillna(0)
        
        logger.info(f"Prepared {len(X)} training samples with {len(X.columns)} features")
        return X, y
    
    async def _save_model(self, model, name: str, metadata: Dict[str, Any]):
        """Save model and metadata"""
        model_path = self.models_dir / f"{name}.pkl"
        metadata_path = self.models_dir / f"{name}.json"
        
        # Add timestamp to metadata
        metadata["created_at"] = datetime.now().isoformat()
        metadata["model_path"] = str(model_path)
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.model_metadata[name] = metadata
        logger.info(f"Saved model: {name}")
    
    async def train_new_model(self, model_type: str, hyperparameters: Dict[str, Any], 
                            training_size: int = 10000):
        """Train a new model with specified parameters"""
        logger.info(f"Training new {model_type} model...")
        
        try:
            # Generate training data
            training_data = await self.data_processor.generate_training_data(size=training_size)
            X, y = await self._prepare_training_data(training_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Create and train model based on type
            if model_type == "ensemble":
                model = EnsembleIPLPredictor()
                await model.train(X_train, y_train, X_val, y_val)
                
            elif model_type == "deep_learning_tensorflow":
                model = DeepLearningIPLModel(model_type='tensorflow')
                epochs = hyperparameters.get('epochs', 50)
                await model.train(X_train, y_train, X_val, y_val, epochs=epochs)
                
            elif model_type == "deep_learning_pytorch":
                model = DeepLearningIPLModel(model_type='pytorch')
                epochs = hyperparameters.get('epochs', 50)
                await model.train(X_train, y_train, X_val, y_val, epochs=epochs)
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Save model
            model_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await self._save_model(model, model_name, {
                "model_type": model_type,
                "hyperparameters": hyperparameters,
                "training_size": training_size,
                "features": list(X.columns)
            })
            
            self.models[model_name] = model
            logger.info(f"Successfully trained and saved {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to train {model_type} model: {e}")
            raise
    
    def get_model(self, model_name: str):
        """Get a specific model"""
        return self.models.get(model_name)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with metadata"""
        models_info = []
        
        for name, metadata in self.model_metadata.items():
            model_info = {
                "name": name,
                "type": metadata.get("model_type", "unknown"),
                "created_at": metadata.get("created_at", "unknown"),
                "training_size": metadata.get("training_size", 0),
                "is_loaded": name in self.models
            }
            models_info.append(model_info)
        
        return models_info
    
    async def delete_model(self, model_name: str):
        """Delete a model and its files"""
        try:
            # Remove from memory
            if model_name in self.models:
                del self.models[model_name]
            
            if model_name in self.model_metadata:
                del self.model_metadata[model_name]
            
            # Remove files
            model_file = self.models_dir / f"{model_name}.pkl"
            metadata_file = self.models_dir / f"{model_name}.json"
            
            if model_file.exists():
                model_file.unlink()
            
            if metadata_file.exists():
                metadata_file.unlink()
            
            logger.info(f"Deleted model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            raise
    
    def get_best_model(self) -> Optional[Any]:
        """Get the best performing model"""
        # For now, return ensemble if available, otherwise first available model
        if "ensemble_predictor" in self.models:
            return self.models["ensemble_predictor"]
        elif self.models:
            return list(self.models.values())[0]
        else:
            return None