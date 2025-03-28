import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from typing import List, Dict
from datetime import datetime

class SupplyChainCalendar(AbstractHolidayCalendar):
    """Custom holiday calendar for global supply chain"""
    rules = [
        Holiday("New Year's Day", month=1, day=1),
        Holiday("Christmas Day", month=12, day=25),
        Holiday("Global Logistics Day", month=10, day=20)
    ]

class DateUtils:
    """Date/time utilities for supply chain forecasting"""
    
    @staticmethod
    def create_datetime_features(df: pd.DataFrame, 
                                date_col: str = 'date') -> pd.DataFrame:
        """Generate time series features from datetime"""
        df['hour'] = df[date_col].dt.hour
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['quarter'] = df[date_col].dt.quarter
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['is_holiday'] = df[date_col].isin(
            SupplyChainCalendar().holidays(start=df[date_col].min(), 
                                        end=df[date_col].max())
        return df
    
    @staticmethod
    def get_fiscal_weeks(year: int, 
                        start_month: int = 4) -> Dict[datetime, int]:
        """Generate fiscal weeks for supply chain planning"""
        dates = pd.date_range(
            start=f'{year}-{start_month}-01',
            end=f'{year+1}-{start_month-1}-30',
            freq='W'
        )
        return {d: i+1 for i, d in enumerate(dates)}
    
    @staticmethod
    def calculate_lead_time(start_dates: pd.Series, 
                           end_dates: pd.Series) -> pd.Series:
        """Calculate business days between two date series"""
        return pd.DataFrame({'start': start_dates, 'end': end_dates}) \
            .apply(lambda x: np.busday_count(x['start'].date(), x['end'].date()), axis=1)
