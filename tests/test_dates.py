import pytest
from datetime import datetime
from utils.date_utils import DateUtils, SupplyChainCalendar

def test_holiday_detection():
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=365)
    })
    df = DateUtils.create_datetime_features(df)
    
    holidays = df[df['is_holiday']]
    assert datetime(2023,10,20) in holidays['date'].values
    assert len(holidays) == 3  # New Year, Christmas, Logistics Day

def test_fiscal_weeks():
    weeks = DateUtils.get_fiscal_weeks(2023, start_month=4)
    assert weeks[datetime(2023,4,3)] == 1
    assert weeks[datetime(2024,3,25)] == 52

def test_lead_time():
    starts = pd.Series([datetime(2023,1,1), datetime(2023,1,5)])
    ends = pd.Series([datetime(2023,1,10), datetime(2023,1,7)])
    
    lead_times = DateUtils.calculate_lead_time(starts, ends)
    assert lead_times.tolist() == [6, 1]  # Weekend exclusion
