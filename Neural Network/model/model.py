"""
Updated Model for Sleep Quality Regression with Target Normalization.
Overwrite your existing model.py with this content.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras import regularizers

set_random_seed(42)

def _sanitize_X(X):
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

class WeightChangeLogger(Callback):
    def on_train_begin(self, logs=None):
        self.prev_weights = [w.numpy().copy() for w in self.model.weights]
    def on_epoch_end(self, epoch, logs=None):
        self.prev_weights = [w.numpy().copy() for w in self.model.weights]

class Model:
    def __init__(self, input_size):
        self.input_size = input_size
        self.scaler = StandardScaler()
        self._scaler_fitted = False
        
        # Scaling factor for the target (e.g., Sleep Quality 0-100 -> divide by 100)
        self.target_scale = 100.0 

        self.model = Sequential([                
            Dense(128, activation='relu', input_shape=(input_size,), kernel_regularizer=regularizers.l2(1e-4)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu', input_shape=(input_size,), kernel_regularizer=regularizers.l2(1e-4)),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(16, activation='relu'),
            
            Dense(1, activation='sigmoid') # Sigmoid forces output 0.0 - 1.0 (perfect for normalized score)
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='mean_squared_error',
            metrics=['mae'],
            run_eagerly=False
        )

    def fit_scaler(self, X):
        X = _sanitize_X(X)
        self.scaler.fit(X)
        self._scaler_fitted = True

    def _ensure_scaler(self, X):
        if not self._scaler_fitted:
            self.fit_scaler(X)

    def fit(self, X, y, epochs=30, batch_size=16, validation_split=0.2, leak_corr_threshold=None):
        X = _sanitize_X(X)
        y = np.asarray(y, dtype=np.float32)
        
        # Split
        groups = np.arange(X.shape[0])
        gss = GroupShuffleSplit(n_splits=1, test_size=validation_split, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups=groups))
        
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        # Fit Scaler
        if not self._scaler_fitted:
            self.fit_scaler(X_tr)
            
        X_tr_scaled = self.scaler.transform(X_tr)
        X_va_scaled = self.scaler.transform(X_va)
        
        # NORMALIZE TARGETS (Critical Step)
        y_tr_norm = y_tr / self.target_scale
        y_va_norm = y_va / self.target_scale

        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1)

        print(f"Training on {len(X_tr)} samples (Target Normalized 0-1)...")
        
        history = self.model.fit(
            X_tr_scaled, y_tr_norm,
            validation_data=(X_va_scaled, y_va_norm),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        return history

    def evaluate(self, X, y):
        X = _sanitize_X(X)
        y = np.asarray(y, dtype=np.float32)
        self._ensure_scaler(X)
        
        X_scaled = self.scaler.transform(X)
        
        # Predict normalized values (0-1)
        preds_norm = self.model.predict(X_scaled, verbose=0)
        
        # Scale back to original range (0-100)
        preds_original = preds_norm * self.target_scale
        
        # Calculate Real Metrics manually
        mse = np.mean((preds_original.flatten() - y) ** 2)
        mae = np.mean(np.abs(preds_original.flatten() - y))
        
        print(f"[eval] Real Scale Metrics -> MSE: {mse:.2f}, MAE: {mae:.2f}")
        
        # Return metrics to Flower
        # (loss, num_examples, dictionary)
        return float(mse), float(mae), 0

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

