import pandas as pd
import numpy as np
import requests
from pydantic import BaseModel, ValidationError
from typing import Union, Dict, Any
from pathlib import Path
import json

class DatasetSchema(BaseModel):
    required_columns: list
    date_column: str
    numeric_columns: list

class DataLoader:
    """Unified data loading utility for supply chain data"""
    
    def __init__(self, config_path: str = "config/data_config.json"):
        self.config = self._load_config(config_path)
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path) as f:
            return json.load(f)
    
    def load_dataset(self, source: str) -> pd.DataFrame:
        """Load data from configured source"""
        source_config = self.config.get(source)
        if not source_config:
            raise ValueError(f"Source {source} not configured")
            
        loader = getattr(self, f"_load_{source_config['type']}")
        df = loader(source_config)
        self._validate_data(df, source_config['schema'])
        return df
    
    def _load_csv(self, config: dict) -> pd.DataFrame:
        return pd.read_csv(
            config['path'],
            parse_dates=config.get('parse_dates', []),
            dtype=config.get('dtypes', {})
        )
    
    def _load_parquet(self, config: dict) -> pd.DataFrame:
        return pd.read_parquet(config['path'])
    
    def _load_api(self, config: dict) -> pd.DataFrame:
        response = requests.get(
            config['url'],
            params=config.get('params', {}),
            headers=config.get('headers', {})
        response.raise_for_status()
        return pd.DataFrame(response.json()['data'])
    
    def _validate_data(self, df: pd.DataFrame, schema_name: str):
        schema = DatasetSchema(**self.config['schemas'][schema_name])
        
        # Check required columns
        missing = set(schema.required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
            
        # Validate date format
        try:
            pd.to_datetime(df[schema.date_column])
        except Exception as e:
            raise ValueError(f"Date validation failed: {str(e)}")
            
        # Check numeric columns
        non_numeric = df[schema.numeric_columns].select_dtypes(exclude=np.number).columns.tolist()
        if non_numeric:
            raise ValueError(f"Non-numeric columns: {non_numeric}")
