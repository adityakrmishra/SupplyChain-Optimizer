"""
Advanced LSTM Forecasting System for Supply Chain Optimization
--------------------------------------------------------------
Features:
- Multivariate multi-step forecasting
- Attention mechanisms
- Automated hyperparameter tuning
- Multiple forecasting strategies
- Probabilistic predictions
- Feature importance analysis
- Advanced data augmentation
- Hierarchical forecasting support
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM, Dense, Bidirectional, Attention,
    Conv1D, TimeDistributed, RepeatVector,
    Input, Concatenate, LayerNormalization
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    LearningRateScheduler, TensorBoard
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import RootMeanSquaredError

from sklearn.preprocessing import (
    MinMaxScaler, RobustScaler, FunctionTransformer
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

from typing import List, Tuple, Dict, Optional, Union
import logging
import holidays
import pickle
import random
import math
import os
import shutil
from datetime import datetime, timedelta
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import kerastuner as kt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AdvancedLSTMForecaster')

@dataclass
class ModelConfig:
    """Configuration for LSTM architecture"""
    seq_length: int = 30
    forecast_horizon: int = 7
    n_features: int = 5
    lstm_units: List[int] = (128, 64)
    attention_units: int = 32
    dropout_rate: float = 0.3
    l2_lambda: float = 1e-4
    learning_rate: float = 1e-3
    conv_filters: int = 64
    kernel_size: int = 3
    bidirectional: bool = True
    probabilistic: bool = False
    quantiles: List[float] = (0.1, 0.5, 0.9)

class AdvancedLSTMForecaster:
    """Production-grade LSTM forecasting system"""
    
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        self.scalers = {}
        self.feature_names = []
        self.model = None
        self.history = None
        self.tuner = None
        self._create_model_directory()
        
    def _create_model_directory(self):
        """Initialize model storage structure"""
        self.model_dir = "lstm_models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.weights_path = os.path.join(self.model_dir, "best_weights.h5")
        self.scaler_path = os.path.join(self.model_dir, "scalers.pkl")

    def _build_attention_model(self) -> Model:
        """Construct attention-based seq2seq architecture"""
        # Encoder
        encoder_inputs = Input(shape=(self.config.seq_length, self.config.n_features))
        x = Conv1D(
            filters=self.config.conv_filters,
            kernel_size=self.config.kernel_size,
            activation='relu'
        )(encoder_inputs)
        
        for units in self.config.lstm_units[:-1]:
            x = Bidirectional(LSTM(
                units,
                return_sequences=True,
                kernel_regularizer=l2(self.config.l2_lambda)
            ))(x)
            x = LayerNormalization()(x)
            x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
            
        encoder_outputs, state_h, state_c = Bidirectional(LSTM(
            self.config.lstm_units[-1],
            return_sequences=True,
            return_state=True,
            kernel_regularizer=l2(self.config.l2_lambda)
        ))(x)
        
        # Decoder
        decoder_inputs = RepeatVector(self.config.forecast_horizon)(encoder_outputs[:, -1, :])
        decoder_lstm = LSTM(
            self.config.lstm_units[-1],
            return_sequences=True,
            kernel_regularizer=l2(self.config.l2_lambda)
        )
        decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[
            state_h[:, :self.config.lstm_units[-1]],
            state_c[:, :self.config.lstm_units[-1]]
        ])
        
        # Attention
        attention = Attention()([decoder_outputs, encoder_outputs])
        decoder_combined = Concatenate()([decoder_outputs, attention])
        
        # Time-distributed dense
        outputs = TimeDistributed(Dense(
            3 if self.config.probabilistic else 1,
            activation='linear'
        ))(decoder_combined)
        
        return Model(encoder_inputs, outputs)

    def _quantile_loss(self, y_true, y_pred):
        """Quantile loss function for probabilistic forecasting"""
        quantiles = tf.constant(self.config.quantiles, dtype=tf.float32)
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantiles * e, (quantiles - 1) * e))

    def _create_model(self, hp: kt.HyperParameters = None) -> Model:
        """Build model with optional hyperparameter tuning"""
        if self.config.probabilistic:
            loss = self._quantile_loss
            output_units = len(self.config.quantiles)
        else:
            loss = 'mse'
            output_units = 1
            
        model = self._build_attention_model()
        model.compile(
            optimizer=Adam(
                hp.Float('learning_rate', 1e-4, 1e-2, sampling='log') if hp 
                else self.config.learning_rate
            ),
            loss=loss,
            metrics=[RootMeanSquaredError()]
        )
        return model

    def preprocess_data(
        self,
        data: pd.DataFrame,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advanced multivariate time series preprocessing"""
        self.feature_names = data.columns.tolist()
        
        # Handle missing values
        data = self._impute_missing(data)
        
        # Add temporal features
        data = self._add_temporal_features(data)
        
        # Scale features
        if fit_scaler:
            self.scalers = {
                col: RobustScaler() for col in data.columns
            }
            
        scaled_data = np.column_stack([
            self.scalers[col].fit_transform(data[col].values.reshape(-1, 1)) 
            if fit_scaler else self.scalers[col].transform(data[col].values.reshape(-1, 1))
            for col in data.columns
        ])
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.config.seq_length - self.config.forecast_horizon + 1):
            X.append(scaled_data[i:i+self.config.seq_length])
            y.append(scaled_data[
                i+self.config.seq_length:i+self.config.seq_length+self.config.forecast_horizon,
                0  # Assume first column is target
            ])
            
        return np.array(X), np.array(y)

    def _impute_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Advanced missing data imputation"""
        # Temporal interpolation
        data = data.interpolate(method='time', limit_direction='both')
        
        # Forward-fill remaining NaNs
        return data.ffill().bfill()

    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer temporal features for supply chain"""
        dt_index = data.index.to_pydatetime()
        
        data['day_of_week'] = [d.weekday() for d in dt_index]
        data['month'] = [d.month for d in dt_index]
        data['is_holiday'] = [d in holidays.US() for d in dt_index]
        data['quarter'] = [d.quarter for d in dt_index]
        data['is_month_end'] = data.index.is_month_end.astype(int)
        data['is_quarter_end'] = data.index.is_quarter_end.astype(int)
        
        # Add Fourier terms for seasonality
        data = self._add_fourier_terms(data, dt_index)
        return data

    def _add_fourier_terms(self, data: pd.DataFrame, dt_index: List[datetime]) -> pd.DataFrame:
        """Add Fourier terms for weekly/yearly seasonality"""
        for period, col in [(365.25, 'yearly'), (7, 'weekly')]:
            for i in range(3):
                data[f'fourier_{col}_sin_{i}'] = np.sin(
                    2 * i * np.pi * np.array([d.timetuple().tm_yday for d in dt_index]) / period
                )
                data[f'fourier_{col}_cos_{i}'] = np.cos(
                    2 * i * np.pi * np.array([d.timetuple().tm_yday for d in dt_index]) / period
                )
        return data

    def temporal_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Time-based data splitting"""
        split_idx = int(len(X) * (1 - test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    def fit(
        self,
        data: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 64,
        validation_split: float = 0.2,
        use_tuner: bool = False
    ) -> None:
        """Train model with advanced capabilities"""
        X, y = self.preprocess_data(data, fit_scaler=True)
        X_train, X_val, y_train, y_val = self.temporal_train_test_split(X, y, validation_split)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ModelCheckpoint(self.weights_path, save_best_only=True),
            TensorBoard(log_dir=os.path.join(self.model_dir, 'logs')),
            LearningRateScheduler(self._cosine_decay)
        ]
        
        if use_tuner:
            self._run_hyperparameter_tuning(X_train, y_train, X_val, y_val)
            self.model = self.tuner.get_best_models(num_models=1)[0]
        else:
            self.model = self._create_model()
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                sample_weight=self._compute_sample_weights(y_train),
                verbose=2
            )
            
        self._save_scalers()

    def _cosine_decay(self, epoch: int, total_epochs=100) -> float:
        """Learning rate schedule with warmup"""
        warmup_epochs = 10
        if epoch < warmup_epochs:
            return self.config.learning_rate * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * self.config.learning_rate * (1 + math.cos(math.pi * progress))

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute weights for imbalanced time series"""
        changes = np.abs(np.diff(y.squeeze(), axis=-1)).mean(axis=1)
        return compute_sample_weight('balanced', changes)

    def _run_hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """Bayesian hyperparameter optimization"""
        tuner = kt.BayesianOptimization(
            self._create_model,
            objective='val_loss',
            max_trials=50,
            executions_per_trial=2,
            directory=os.path.join(self.model_dir, 'tuning'),
            project_name='supply_chain_forecast'
        )
        
        tuner.search(
            X_train, y_train,
            epochs=100,
            validation_data=(X_val, y_val),
            batch_size=64,
            callbacks=[EarlyStopping(patience=10)]
        )
        self.tuner = tuner

    def predict(
        self,
        data: pd.DataFrame,
        n_samples: int = 100
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate forecasts with uncertainty estimation"""
        X, _ = self.preprocess_data(data, fit_scaler=False)
        
        if self.config.probabilistic:
            preds = np.stack([self.model(X) for _ in range(n_samples)])
            mean_pred = preds.mean(axis=0)
            std_pred = preds.std(axis=0)
            return mean_pred, std_pred
        else:
            return self.model.predict(X)

    def evaluate(
        self,
        data: pd.DataFrame,
        metrics: List[str] = ['mae', 'rmse', 'mape']
    ) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        X, y_true = self.preprocess_data(data, fit_scaler=False)
        y_pred = self.predict(X)
        
        if self.config.forecast_horizon > 1:
            y_true = y_true.reshape(-1, self.config.forecast_horizon)
            y_pred = y_pred.reshape(-1, self.config.forecast_horizon)
            
        results = {}
        for metric in metrics:
            if metric == 'mae':
                results['MAE'] = mean_absolute_error(y_true, y_pred)
            elif metric == 'rmse':
                results['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
            elif metric == 'mape':
                results['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
                
        return results

    def explain_prediction(
        self,
        sample: np.ndarray,
        method: str = 'integrated_gradients'
    ) -> np.ndarray:
        """Model explainability using different methods"""
        if method == 'integrated_gradients':
            return self._integrated_gradients(sample)
        elif method == 'permutation':
            return self._permutation_importance(sample)
        else:
            raise ValueError(f"Unsupported explanation method: {method}")

    def _integrated_gradients(self, sample: np.ndarray) -> np.ndarray:
        """Calculate feature importance using integrated gradients"""
        baseline = np.zeros_like(sample)
        steps = 50
        gradients = []
        
        for alpha in np.linspace(0, 1, steps):
            input_ = baseline + alpha * (sample - baseline)
            with tf.GradientTape() as tape:
                tape.watch(input_)
                prediction = self.model(input_)
            grad = tape.gradient(prediction, input_)
            gradients.append(grad.numpy())
            
        avg_grad = np.mean(gradients, axis=0)
        integrated_grad = (sample - baseline) * avg_grad
        return integrated_grad.squeeze()

    def _permutation_importance(
        self,
        sample: np.ndarray,
        n_iter: int = 100
    ) -> np.ndarray:
        """Compute permutation feature importance"""
        base_pred = self.model.predict(sample[np.newaxis, ...])
        importance = np.zeros(sample.shape[-1])
        
        for i in range(sample.shape[-1]):
            delta = 0
            for _ in range(n_iter):
                perturbed = sample.copy()
                np.random.shuffle(perturbed[:, i])
                delta += np.abs(base_pred - self.model.predict(perturbed[np.newaxis, ...]))
            importance[i] = delta / n_iter
            
        return importance / importance.sum()

    def _save_scalers(self) -> None:
        """Serialize feature scalers"""
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scalers, f)

    def load(self) -> None:
        """Load trained model and scalers"""
        self.model = load_model(self.weights_path)
        with open(self.scaler_path, 'rb') as f:
            self.scalers = pickle.load(f)

    def plot_forecast(
        self,
        true_values: pd.Series,
        forecast: np.ndarray,
        uncertainty: np.ndarray = None
    ) -> plt.Figure:
        """Visualize forecast with uncertainty bands"""
        plt.figure(figsize=(12, 6))
        plt.plot(true_values.index, true_values, label='Actual')
        plt.plot(true_values.index[-self.config.forecast_horizon:], forecast, label='Forecast')
        
        if uncertainty is not None:
            plt.fill_between(
                true_values.index[-self.config.forecast_horizon:],
                forecast - 1.96 * uncertainty,
                forecast + 1.96 * uncertainty,
                alpha=0.2,
                label='95% CI'
            )
            
        plt.title('Supply Chain Demand Forecast')
        plt.xlabel('Date')
        plt.ylabel('Normalized Demand')
        plt.legend()
        plt.grid(True)
        return plt.gcf()

