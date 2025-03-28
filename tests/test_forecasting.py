import pytest
import pandas as pd
from unittest.mock import Mock
from datetime import datetime
from forecasting.prophet_model import ProphetForecaster
from forecasting.lstm_model import LSTMForecaster
from utils.data_loader import DataLoader, DatasetSchema

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
    return pd.DataFrame({
        'ds': dates,
        'y': np.sin(np.linspace(0, 20, len(dates))) * 100 + 50
    })

@pytest.fixture
def prophet_model():
    return ProphetForecaster(growth='flat', seasonality_mode='additive')

@pytest.fixture
def lstm_model():
    model = LSTMForecaster(sequence_length=7)
    model.model = Mock()  # Mock actual LSTM for faster testing
    return model

def test_prophet_forecast(sample_data, prophet_model):
    prophet_model.fit(sample_data)
    forecast = prophet_model.predict(periods=30)
    
    assert isinstance(forecast, pd.DataFrame)
    assert all(col in forecast.columns for col in ['ds', 'yhat', 'yhat_lower', 'yhat_upper'])
    assert len(forecast) == 30
    assert forecast['yhat'].isna().sum() == 0

def test_lstm_predictions(sample_data, lstm_model):
    lstm_model.fit(sample_data['y'])
    predictions = lstm_model.predict(sample_data['y'], forecast_steps=14)
    
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 14
    assert not np.isnan(predictions).any()

def test_data_loader_schema():
    loader = DataLoader(config_path="tests/test_data_config.json")
    test_data = loader.load_dataset('test_csv')
    
    assert DatasetSchema(**loader.config['schemas']['test_schema']).required_columns
    assert 'date' in test_data.columns
    assert test_data['value'].dtype == np.float64

def test_model_evaluation(sample_data, prophet_model):
    prophet_model.fit(sample_data)
    forecast = prophet_model.predict(30)
    eval_metrics = prophet_model.evaluate(sample_data[-30:], forecast)
    
    assert 'MAE' in eval_metrics
    assert 'RMSE' in eval_metrics
    assert eval_metrics['MAE'] > 0
