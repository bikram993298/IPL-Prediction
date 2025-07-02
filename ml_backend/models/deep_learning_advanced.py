"""
Advanced Deep Learning Models with TensorFlow and PyTorch
Implements LSTM, Transformer, and CNN architectures for IPL prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from typing import Dict, Any, Tuple, List
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)

class AdvancedTensorFlowModel:
    """Advanced TensorFlow model with multiple architectures"""
    
    def __init__(self, model_type='transformer'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.history = None
        
    def _build_transformer_model(self, input_shape: Tuple[int, ...]):
        """Build Transformer model for sequence prediction"""
        inputs = layers.Input(shape=input_shape)
        
        # Embedding and positional encoding
        x = layers.Dense(128)(inputs)
        x = layers.Dropout(0.1)(x)
        
        # Multi-head attention layers
        for _ in range(3):
            # Multi-head attention
            attention = layers.MultiHeadAttention(
                num_heads=8,
                key_dim=64,
                dropout=0.1
            )(x, x)
            
            # Add & Norm
            x = layers.Add()([x, attention])
            x = layers.LayerNormalization()(x)
            
            # Feed forward
            ff = layers.Dense(256, activation='relu')(x)
            ff = layers.Dropout(0.1)(ff)
            ff = layers.Dense(128)(ff)
            
            # Add & Norm
            x = layers.Add()([x, ff])
            x = layers.LayerNormalization()(x)
        
        # Global pooling and output
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def _build_lstm_model(self, input_shape: Tuple[int, ...]):
        """Build LSTM model for time series prediction"""
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    async def train(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series, epochs: int = 100):
        """Train the deep learning model"""
        logger.info(f"🧠 Training {self.model_type} TensorFlow model...")
        
        self.feature_names = list(X_train.columns)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Reshape for sequence models
        if self.model_type in ['lstm', 'transformer']:
            X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
            X_val_scaled = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
        
        # Build model
        input_shape = X_train_scaled.shape[1:]
        
        if self.model_type == 'transformer':
            self.model = self._build_transformer_model(input_shape)
        elif self.model_type == 'lstm':
            self.model = self._build_lstm_model(input_shape)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['mae', 'mse']
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        logger.info(f"✅ {self.model_type} model training completed!")
        
        return self.history
    
    async def predict(self, features: Dict[str, Any]) -> float:
        """Make prediction using the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Convert features to array
        X = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0.0
        
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Reshape for sequence models
        if self.model_type in ['lstm', 'transformer']:
            X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        prediction = self.model.predict(X_scaled, verbose=0)[0][0]
        return float(np.clip(prediction, 0.01, 0.99))

class AdvancedPyTorchModel:
    """Advanced PyTorch model with custom architectures"""
    
    def __init__(self, model_type='transformer'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    class TransformerNet(nn.Module):
        def __init__(self, input_size, d_model=128, nhead=8, num_layers=3):
            super().__init__()
            self.input_projection = nn.Linear(input_size, d_model)
            self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            
            self.classifier = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            # x shape: (batch_size, seq_len, input_size)
            x = self.input_projection(x)
            
            # Add positional encoding
            seq_len = x.size(1)
            x = x + self.positional_encoding[:seq_len].unsqueeze(0)
            
            # Transformer
            x = self.transformer(x)
            
            # Global average pooling
            x = x.mean(dim=1)
            
            # Classification
            return self.classifier(x)
    
    class LSTMNet(nn.Module):
        def __init__(self, input_size, hidden_size=128, num_layers=3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=0.3
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            lstm_out, (hidden, _) = self.lstm(x)
            # Use the last hidden state
            return self.classifier(hidden[-1])
    
    async def train(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series, epochs: int = 100):
        """Train the PyTorch model"""
        logger.info(f"🔥 Training {self.model_type} PyTorch model...")
        
        self.feature_names = list(X_train.columns)
        input_size = len(self.feature_names)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)  # Add sequence dimension
        y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val_scaled).unsqueeze(1)
        y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1)
        
        # Initialize model
        if self.model_type == 'transformer':
            self.model = self.TransformerNet(input_size)
        elif self.model_type == 'lstm':
            self.model = self.LSTMNet(input_size)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, verbose=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            X_train_batch = X_train_tensor.to(self.device)
            y_train_batch = y_train_tensor.to(self.device)
            
            outputs = self.model(X_train_batch)
            loss = criterion(outputs, y_train_batch)
            loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                X_val_batch = X_val_tensor.to(self.device)
                y_val_batch = y_val_tensor.to(self.device)
                
                val_outputs = self.model(X_val_batch)
                val_loss = criterion(val_outputs, y_val_batch)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
        
        self.is_trained = True
        logger.info(f"✅ {self.model_type} PyTorch model training completed!")
    
    async def predict(self, features: Dict[str, Any]) -> float:
        """Make prediction using the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Convert features to array
        X = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0.0
        
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X_tensor).cpu().numpy()[0][0]
        
        return float(np.clip(prediction, 0.01, 0.99))

class DeepLearningEnsemble:
    """Ensemble of multiple deep learning models"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
    async def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series):
        """Train ensemble of deep learning models"""
        logger.info("🚀 Training Deep Learning Ensemble...")
        
        model_configs = [
            ('tf_transformer', AdvancedTensorFlowModel('transformer')),
            ('tf_lstm', AdvancedTensorFlowModel('lstm')),
            ('pytorch_transformer', AdvancedPyTorchModel('transformer')),
            ('pytorch_lstm', AdvancedPyTorchModel('lstm'))
        ]
        
        val_scores = {}
        
        for name, model in model_configs:
            try:
                logger.info(f"Training {name}...")
                await model.train(X_train, y_train, X_val, y_val, epochs=50)
                
                # Evaluate
                predictions = []
                for _, row in X_val.iterrows():
                    pred = await model.predict(row.to_dict())
                    predictions.append(pred)
                
                # Calculate R² score
                from sklearn.metrics import r2_score
                score = r2_score(y_val, predictions)
                val_scores[name] = score
                self.models[name] = model
                
                logger.info(f"✅ {name} - R² Score: {score:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error training {name}: {e}")
                val_scores[name] = 0.0
        
        # Calculate weights
        self._calculate_weights(val_scores)
        self.is_trained = True
        
        logger.info("🎯 Deep Learning Ensemble training completed!")
    
    def _calculate_weights(self, val_scores: Dict[str, float]):
        """Calculate ensemble weights"""
        # Filter out poor models
        filtered_scores = {k: v for k, v in val_scores.items() if v > 0.1}
        
        if not filtered_scores:
            self.weights = {k: 1/len(val_scores) for k in val_scores.keys()}
            return
        
        # Softmax weighting
        scores = np.array(list(filtered_scores.values()))
        exp_scores = np.exp(scores * 3)
        weights = exp_scores / np.sum(exp_scores)
        
        self.weights = {}
        for i, model_name in enumerate(filtered_scores.keys()):
            self.weights[model_name] = weights[i]
        
        # Zero weight for poor models
        for model_name in val_scores.keys():
            if model_name not in filtered_scores:
                self.weights[model_name] = 0.0
        
        logger.info(f"🎯 Deep Learning weights: {self.weights}")
    
    async def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Make ensemble prediction"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        predictions = []
        total_weight = 0
        
        for name, model in self.models.items():
            if name in self.weights and self.weights[name] > 0:
                try:
                    pred = await model.predict(features)
                    weight = self.weights[name]
                    predictions.append(pred * weight)
                    total_weight += weight
                except Exception as e:
                    logger.warning(f"Prediction error for {name}: {e}")
        
        if total_weight == 0:
            return {'team1_win_prob': 0.5, 'team2_win_prob': 0.5}
        
        ensemble_pred = sum(predictions) / total_weight
        ensemble_pred = np.clip(ensemble_pred, 0.01, 0.99)
        
        return {
            'team1_win_prob': ensemble_pred,
            'team2_win_prob': 1.0 - ensemble_pred
        }