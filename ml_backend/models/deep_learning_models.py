"""
Deep Learning Models for IPL Win Prediction
Advanced neural networks using TensorFlow/Keras and PyTorch
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class DeepLearningIPLModel:
    """Advanced deep learning model for IPL predictions"""
    
    def __init__(self, model_type='tensorflow'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
        if model_type == 'tensorflow':
            self._build_tensorflow_model()
        elif model_type == 'pytorch':
            self._build_pytorch_model()
    
    def _build_tensorflow_model(self):
        """Build TensorFlow/Keras neural network"""
        self.model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(None,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'mae']
        )
        
        logger.info("TensorFlow model built successfully")
    
    def _build_pytorch_model(self):
        """Build PyTorch neural network"""
        class IPLNet(nn.Module):
            def __init__(self, input_size):
                super(IPLNet, self).__init__()
                self.fc1 = nn.Linear(input_size, 256)
                self.bn1 = nn.BatchNorm1d(256)
                self.dropout1 = nn.Dropout(0.3)
                
                self.fc2 = nn.Linear(256, 128)
                self.bn2 = nn.BatchNorm1d(128)
                self.dropout2 = nn.Dropout(0.3)
                
                self.fc3 = nn.Linear(128, 64)
                self.bn3 = nn.BatchNorm1d(64)
                self.dropout3 = nn.Dropout(0.2)
                
                self.fc4 = nn.Linear(64, 32)
                self.dropout4 = nn.Dropout(0.2)
                
                self.fc5 = nn.Linear(32, 1)
                
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                x = self.dropout1(self.bn1(self.relu(self.fc1(x))))
                x = self.dropout2(self.bn2(self.relu(self.fc2(x))))
                x = self.dropout3(self.bn3(self.relu(self.fc3(x))))
                x = self.dropout4(self.relu(self.fc4(x)))
                x = self.sigmoid(self.fc5(x))
                return x
        
        # Will be initialized after we know input size
        self.model_class = IPLNet
        logger.info("PyTorch model class defined")
    
    async def train(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series, epochs: int = 100):
        """Train the deep learning model"""
        logger.info(f"Training {self.model_type} deep learning model...")
        
        self.feature_names = list(X_train.columns)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        if self.model_type == 'tensorflow':
            await self._train_tensorflow(X_train_scaled, y_train, X_val_scaled, y_val, epochs)
        elif self.model_type == 'pytorch':
            await self._train_pytorch(X_train_scaled, y_train, X_val_scaled, y_val, epochs)
        
        self.is_trained = True
        logger.info("Deep learning model training completed!")
    
    async def _train_tensorflow(self, X_train, y_train, X_val, y_val, epochs):
        """Train TensorFlow model"""
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    async def _train_pytorch(self, X_train, y_train, X_val, y_val, epochs):
        """Train PyTorch model"""
        # Initialize model with correct input size
        input_size = X_train.shape[1]
        self.model = self.model_class(input_size)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
    
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
        
        if self.model_type == 'tensorflow':
            prediction = self.model.predict(X_scaled, verbose=0)[0][0]
        elif self.model_type == 'pytorch':
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                prediction = self.model(X_tensor).item()
        
        return float(np.clip(prediction, 0.01, 0.99))

class LSTMTimeSeriesModel:
    """LSTM model for time series prediction of match progression"""
    
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def _build_lstm_model(self, input_shape):
        """Build LSTM model for time series prediction"""
        self.model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            
            layers.LSTM(32),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        sequences = []
        targets = []
        
        # Group by match
        for match_id in data['match_id'].unique():
            match_data = data[data['match_id'] == match_id].sort_values('ball_number')
            
            if len(match_data) >= self.sequence_length:
                for i in range(len(match_data) - self.sequence_length):
                    seq = match_data.iloc[i:i+self.sequence_length][['current_score', 'wickets', 'required_run_rate']].values
                    target = match_data.iloc[i+self.sequence_length]['win_probability']
                    
                    sequences.append(seq)
                    targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    async def train(self, data: pd.DataFrame, epochs: int = 50):
        """Train LSTM model"""
        logger.info("Training LSTM time series model...")
        
        X, y = self.prepare_sequences(data)
        
        # Build model
        self._build_lstm_model((self.sequence_length, X.shape[2]))
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        self.is_trained = True
        return history

class TransformerModel:
    """Transformer model for advanced sequence modeling"""
    
    def __init__(self, d_model=128, num_heads=8, num_layers=4):
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.model = None
        self.is_trained = False
    
    def _build_transformer(self, input_shape):
        """Build transformer model"""
        inputs = layers.Input(shape=input_shape)
        
        # Positional encoding
        x = layers.Dense(self.d_model)(inputs)
        
        # Transformer blocks
        for _ in range(self.num_layers):
            # Multi-head attention
            attention = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads
            )(x, x)
            
            # Add & Norm
            x = layers.Add()([x, attention])
            x = layers.LayerNormalization()(x)
            
            # Feed forward
            ff = layers.Dense(self.d_model * 4, activation='relu')(x)
            ff = layers.Dense(self.d_model)(ff)
            
            # Add & Norm
            x = layers.Add()([x, ff])
            x = layers.LayerNormalization()(x)
        
        # Global average pooling and output
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    async def train(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train transformer model"""
        logger.info("Training Transformer model...")
        
        self._build_transformer(X_train.shape[1:])
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        self.is_trained = True
        return history