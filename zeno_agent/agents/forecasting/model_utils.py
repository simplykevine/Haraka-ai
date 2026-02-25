# zeno_agent/agents/forecasting/model_utils.py
"""
Forecasting model utilities using Prophet for time series prediction.
No circular imports - standalone implementation.
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Tuple, Dict, Any


def prepare_prophet_df(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Convert raw trade data to Prophet-compatible format.
    Expected input: DataFrame with 'ds' (datetime) and target column.
    """
    if 'ds' not in df.columns:
        raise ValueError("Input DataFrame must have 'ds' column (datetime)")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    prophet_df = df[['ds', target_col]].rename(columns={target_col: 'y'})
    prophet_df = prophet_df.dropna(subset=['ds', 'y'])
    return prophet_df


def run_model(
    df: pd.DataFrame,
    periods: int,
    metric_name: str = "unknown"
) -> Tuple[np.ndarray, Dict[str, Any], str]:
    """
    Run Prophet forecasting model on time series data.
    
    Args:
        df: DataFrame with 'ds' (datetime) and 'y' (target) columns
        periods: Number of future periods to forecast
        metric_name: Name of metric for logging/debugging
    
    Returns:
        Tuple of (forecast_values, confidence_intervals, model_type)
    """
    try:
        # Validate input
        if df.empty or len(df) < 4:
            raise ValueError(f"Insufficient data for forecasting ({len(df)} points)")
        
        # Initialize and fit Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95
        )
        model.fit(df)
        
        # Generate future dates and forecast
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        
        # Extract forecast values and confidence intervals
        forecast_values = forecast['yhat'].tail(periods).values
        lower_bound = forecast['yhat_lower'].tail(periods).values
        upper_bound = forecast['yhat_upper'].tail(periods).values
        
        intervals = {
            "lower": lower_bound.tolist(),
            "upper": upper_bound.tolist(),
            "mean": forecast_values.tolist()
        }
        
        return forecast_values, intervals, "prophet"
        
    except Exception as e:
        # Fallback to simple linear extrapolation if Prophet fails
        print(f"[Forecasting] Prophet failed for {metric_name}: {e}. Using linear fallback.")
        
        if len(df) < 2:
            # Not enough points for linear fit - return constant
            last_value = df['y'].iloc[-1] if not df.empty else 0.0
            forecast_values = np.full(periods, last_value)
        else:
            # Simple linear extrapolation
            x = np.arange(len(df))
            y = df['y'].values
            slope, intercept = np.polyfit(x, y, 1)
            future_x = np.arange(len(df), len(df) + periods)
            forecast_values = slope * future_x + intercept
        
        intervals = {
            "lower": (forecast_values * 0.9).tolist(),
            "upper": (forecast_values * 1.1).tolist(),
            "mean": forecast_values.tolist()
        }
        
        return forecast_values, intervals, "linear_fallback"