"""
Advanced Prophet Forecasting for Supply Chain Optimization
----------------------------------------------------------
Features:
- Automated hyperparameter tuning with parallel processing
- Dynamic holiday/event detection
- Advanced preprocessing pipelines
- Multiple accuracy metrics
- Bayesian optimization integration
- Model explainability components
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error,
    mean_absolute_percentage_error
)
import logging
import holidays
from joblib import Parallel, delayed
from hyperopt import fmin, tpe, hp, Trials
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import json
from pathlib import Path

class AdvancedProphetForecaster:
    """Enhanced Prophet model for supply chain forecasting"""
    
    def __init__(self, country: str = 'US', freq: str = 'D', **kwargs):
        """
        Initialize advanced forecaster
        :param country: ISO country code for holiday calendar
        :param freq: Data frequency ('D', 'W', 'M')
        """
        self.model = None
        self.country = country
        self.freq = freq
        self.features = []
        self.logger = logging.getLogger('AdvancedProphetForecaster')
        self._initialize_model(**kwargs)
        
    def _initialize_model(self, **kwargs):
        """Configure Prophet model with supply chain parameters"""
        default_params = {
            'growth': 'logistic',  # Better for supply chain growth patterns
            'seasonality_mode': 'multiplicative',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10,
            'holidays_prior_scale': 10,
            'yearly_seasonality': 8,
            'weekly_seasonality': 3,
            'daily_seasonality': False,
            'uncertainty_samples': 1000
        }
        self.model = Prophet(**{**default_params, **kwargs})
        self._add_holidays()
        self._add_custom_seasonalities()

    def _add_holidays(self):
        """Add country-specific holidays and logistics events"""
        try:
            country_holidays = holidays.CountryHoliday(self.country)
            years = pd.date_range(start='2010', end='2030', freq='Y').year
            holiday_df = pd.DataFrame([
                {'holiday': name, 'ds': date}
                for date, name in country_holidays.years_covered(years).items()
            ])
            
            # Add supply chain specific events
            logistics_events = pd.DataFrame({
                'holiday': 'inventory_day',
                'ds': pd.date_range(start='2010-01-01', end='2030-12-31', freq='Q')
            })
            
            self.model.holidays = pd.concat([holiday_df, logistics_events])
        except Exception as e:
            self.logger.warning(f"Could not load holidays: {str(e)}")

    def _add_custom_seasonalities(self):
        """Add supply chain specific seasonalities"""
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        self.model.add_seasonality(
            name='quarterly',
            period=91.25,
            fourier_order=8
        )

    def preprocess_data(
        self,
        data: pd.DataFrame,
        cap: float = None,
        floor: float = 0
    ) -> pd.DataFrame:
        """
        Enhanced preprocessing pipeline for supply chain data
        :param data: Raw input data with columns ['ds', 'y']
        :param cap: Saturation capacity for logistic growth
        :param floor: Minimum capacity floor
        """
        required_cols = {'ds', 'y'}
        if not required_cols.issubset(data.columns):
            raise ValueError("Missing required columns 'ds' or 'y'")
            
        df = (
            data
            .rename(columns=str.lower)
            .assign(ds=lambda x: pd.to_datetime(x['ds']))
            .sort_values('ds')
            .pipe(self._handle_missing_dates)
            .pipe(self._remove_outliers)
            .assign(cap=cap)
            .assign(floor=floor)
        )
        
        self.features = [c for c in df.columns if c not in ['ds', 'y']]
        for feature in self.features:
            self.model.add_regressor(feature)
            
        return df

    def _handle_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing dates in time series"""
        full_dates = pd.date_range(
            start=df['ds'].min(),
            end=df['ds'].max(),
            freq=self.freq
        )
        return df.set_index('ds').reindex(full_dates).reset_index().ffill()

    def _remove_outliers(self, df: pd.DataFrame, n_std: int = 3) -> pd.DataFrame:
        """Remove statistical outliers"""
        avg = df['y'].mean()
        std = df['y'].std()
        return df[(df['y'] >= avg - n_std*std) & (df['y'] <= avg + n_std*std)]

    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Train model with advanced diagnostics"""
        self.logger.info("Starting model training...")
        self.model.fit(data, **kwargs)
        self.logger.info("Model training completed")
        
        # Store feature importance
        self.feature_importance = self._calculate_feature_importance(data)

    def _calculate_feature_importance(self, data: pd.DataFrame) -> Dict:
        """Calculate permutation feature importance"""
        baseline_score = self.evaluate(data, self.predict(len(data)))
        importance = {}
        
        for col in self.features:
            temp_data = data.copy()
            np.random.shuffle(temp_data[col].values)
            shuffled_score = self.evaluate(temp_data, self.predict(len(data)))
            importance[col] = baseline_score['RMSE'] - shuffled_score['RMSE']
            
        return importance

    def predict(
        self,
        periods: int,
        future_data: pd.DataFrame = None,
        freq: str = None
    ) -> pd.DataFrame:
        """Generate forecast with uncertainty intervals"""
        freq = freq or self.freq
        if future_data is None:
            future = self.model.make_future_dataframe(
                periods=periods,
                freq=freq,
                include_history=False
            )
        else:
            future = future_data
            
        return self.model.predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def cross_validate(
        self,
        horizon: str = '30 days',
        initial: str = '365 days',
        parallel: str = 'processes'
    ) -> pd.DataFrame:
        """Parallelized cross-validation for large datasets"""
        return cross_validation(
            self.model,
            horizon=horizon,
            initial=initial,
            period='30 days',
            parallel=parallel
        )

    def evaluate(
        self,
        true_data: pd.DataFrame,
        forecast: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate comprehensive forecast accuracy metrics"""
        merged = true_data.merge(forecast, on='ds')
        return {
            'MAE': mean_absolute_error(merged['y'], merged['yhat']),
            'RMSE': np.sqrt(mean_squared_error(merged['y'], merged['yhat'])),
            'MAPE': mean_absolute_percentage_error(merged['y'], merged['yhat']),
            'Coverage': ((merged['y'] >= merged['yhat_lower']) & 
                        (merged['y'] <= merged['yhat_upper'])).mean()
        }

    def tune_hyperparameters(
        self,
        data: pd.DataFrame,
        max_evals: int = 100,
        space: Optional[Dict] = None
    ) -> Dict:
        """Bayesian optimization for parameter tuning"""
        default_space = {
            'changepoint_prior_scale': hp.loguniform('changepoint_prior_scale', -5, 0),
            'seasonality_prior_scale': hp.loguniform('seasonality_prior_scale', -5, 0),
            'holidays_prior_scale': hp.loguniform('holidays_prior_scale', -5, 0),
            'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative'])
        }
        space = space or default_space
        
        def objective(params):
            model = Prophet(**params).fit(data)
            df_cv = cross_validation(model, horizon='30 days', parallel='processes')
            return performance_metrics(df_cv)['rmse'].values[0]
            
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )
        return best

    def save_model(self, path: Union[str, Path]):
        """Serialize model to JSON"""
        model_path = Path(path)
        with open(model_path, 'w') as f:
            json.dump(self.model.to_json(), f)

    @classmethod
    def load_model(cls, path: Union[str, Path]):
        """Load serialized model from JSON"""
        model_path = Path(path)
        with open(model_path) as f:
            model_json = json.load(f)
            
        forecaster = cls()
        forecaster.model = Prophet().from_json(model_json)
        return forecaster

    def plot_components(self, forecast: pd.DataFrame):
        """Visualize forecast components with supply chain context"""
        fig = self.model.plot_components(forecast)
        plt.suptitle('Supply Chain Demand Forecast Components', y=1.02)
        plt.tight_layout()
        return fig

# Example Usage
if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    
    # Load sample supply chain data
    data = fetch_openml(name='walmart-sales', parser='auto')
    df = data.frame.rename(columns={'date': 'ds', 'sales': 'y'})
    
    # Initialize and train model
    forecaster = AdvancedProphetForecaster(country='US', freq='W')
    processed_data = forecaster.preprocess_data(df, cap=1e6)
    forecaster.fit(processed_data)
    
    # Generate forecast
    forecast = forecaster.predict(periods=26, freq='W')
    
    # Evaluate performance
    metrics = forecaster.evaluate(processed_data, forecast)
    print(f"Forecast Accuracy: {metrics}")
    
    # Tune hyperparameters
    best_params = forecaster.tune_hyperparameters(processed_data)
    print(f"Optimized Parameters: {best_params}")
    
    # Save model
    forecaster.save_model('supply_chain_prophet.json')
