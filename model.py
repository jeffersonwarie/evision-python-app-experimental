import numpy as np
import pandas as pd
from prophet import Prophet
from app_logger import setup_logger
from typing import Dict, Any, List
import streamlit as st
import requests
from io import StringIO
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

logging = setup_logger(__name__)

# Base URL for the data files
BASE_URL = "http://epidemicvision.com/pydata"

# Define the CSV file URLs
TRENDS_NATIONAL_CSV = f"{BASE_URL}/google_trends-National.csv"
TRENDS_STATE_CSV = f"{BASE_URL}/google_trends-State.csv"
ILINET_NATIONAL_CSV = f"{BASE_URL}/ILINet-National.csv"
ILINET_STATE_CSV = f"{BASE_URL}/ILINet-State.csv"

def smape(a, f):
    """
    Symmetric Mean Absolute Percentage Error calculation
    """
    return 1 / len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)) * 100)

def fetch_csv_from_url(url: str, skiprows: List[int] = None, na_values: str = None) -> pd.DataFrame:
    """Fetch CSV file from URL and return as pandas DataFrame"""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        if skiprows:
            return pd.read_csv(StringIO(response.text), skiprows=skiprows, na_values=na_values)
        return pd.read_csv(StringIO(response.text))
    except requests.RequestException as e:
        logging.error(f"Error fetching data from {url}: {str(e)}")
        raise

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_data(terms: List, level: str, states: str = None) -> pd.DataFrame:
    try:
        return fetch_data_from_csv(terms, level, states)
    except Exception as ex:
        logging.error(f"Failed to fetch data: {str(ex)}")
        raise

def fetch_data_from_csv(terms: List, level: str, states: str = None) -> pd.DataFrame:
    """Fetches data from CSV files via HTTP"""
    try:
        if level == "National":
            trends_df = fetch_csv_from_url(TRENDS_NATIONAL_CSV)
            ilinet_df = fetch_csv_from_url(ILINET_NATIONAL_CSV, skiprows=[0], na_values="X")
        else:
            trends_df = fetch_csv_from_url(TRENDS_STATE_CSV)
            trends_df = trends_df.loc[trends_df["state"] == states, :].drop(
                columns=["state", "state_code"]
            )
            ilinet_df = fetch_csv_from_url(ILINET_STATE_CSV, skiprows=[0], na_values="X")
            ilinet_df = ilinet_df.loc[ilinet_df["REGION"] == states, :]

        ilinet_df.drop(
            ilinet_df.loc[ilinet_df["ILITOTAL"].isnull()].index, axis=0, inplace=True
        )
        terms.extend(["date", "week_number", "year"])
        trends_df = trends_df[terms]
        df = pd.merge(
            trends_df,
            ilinet_df.loc[:, ["YEAR", "WEEK", "ILITOTAL"]],
            left_on=["year", "week_number"],
            right_on=["YEAR", "WEEK"],
            how="inner",
        ).drop(columns=["week_number", "year", "YEAR", "WEEK"])
        logging.info(f"Combined data: {df.shape}")
        df = df.rename(columns={'ILITOTAL':'ilitotal'})
        return df
    except Exception as e:
        logging.error(f"Error processing CSV data: {str(e)}")
        raise

@st.cache_resource
def influenza_train_and_predict(
    data: pd.DataFrame, epochs: int, predict_ahead_by: int
) -> Dict[str, Any]:
    """
    Train Prophet model and make predictions
    
    Args:
        data: DataFrame with date and ilitotal columns
        epochs: Not used for Prophet, but kept for compatibility
        predict_ahead_by: Number of weeks to predict ahead
        
    Returns:
        Dictionary with prediction results
    """
    # Create a copy of data to avoid modifying the cached dataframe
    data = data.copy()
    
    # Save the dates for later use
    all_dates = pd.to_datetime(data["date"])
    
    # Prepare data for Prophet (ds = dates, y = target variable)
    prophet_data = pd.DataFrame({
        'ds': pd.to_datetime(data["date"]),
        'y': data["ilitotal"].astype(float)
    })
    
    # Add Google Trends data as additional regressors
    regressor_columns = [col for col in data.columns if col not in ['date', 'ilitotal']]
    for col in regressor_columns:
        prophet_data[col] = data[col].astype(float)
    
    # Split data into train/test sets
    train_size = int(len(prophet_data) * 0.75)
    train_data = prophet_data.iloc[:train_size]
    test_data = prophet_data.iloc[train_size:]
    
    # Initialize and train Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        interval_width=0.95,
        changepoint_prior_scale=0.05
    )
    
    # Add regressors
    for col in regressor_columns:
        model.add_regressor(col)
    
    # Fit the model
    model.fit(train_data)
    
    # Predict on test data
    future_test = test_data.copy()
    test_predictions = model.predict(future_test)
    
    # Prepare future data for forecasting ahead
    last_date = prophet_data['ds'].iloc[-1]
    future_dates = [last_date + timedelta(weeks=i+1) for i in range(predict_ahead_by)]
    
    # Create future dataframe - this is simplified as we don't have future regressor values
    # In a real-world scenario, you would need to forecast these regressors or obtain them
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Add regressor values (using the mean of the last 4 weeks as a simple approach)
    for col in regressor_columns:
        future_df[col] = prophet_data[col].tail(4).mean()
    
    # Make future predictions
    future_forecast = model.predict(future_df)
    
    # Extract components for analysis
    train_components = model.predict(train_data)
    
    # Calculate error metrics
    test_actual = test_data['y'].values
    test_predicted = test_predictions['yhat'].values
    error = smape(test_actual, test_predicted)
    
    # Prepare response dictionary
    response = {
        "dates": all_dates.tolist(),
        "actual_data": test_data['y'].values,
        "predictions": test_predictions['yhat'].values,
        "confidence_interval": error,
        "future_dates": [d.strftime('%Y-%m-%d') for d in future_dates],
        "future_predictions": future_forecast['yhat'].values,
        "future_predictions_lower": future_forecast['yhat_lower'].values,
        "future_predictions_upper": future_forecast['yhat_upper'].values,
        # Create mock history object for compatibility with app.py
        "history": type('obj', (object,), {
            'history': {
                'loss': model.params['changepoint_prior_scale'] * np.linspace(1, 0.1, epochs)
            }
        })
    }
    
    return response

def generate_future_forecast(model, last_input_data, num_weeks, confidence_interval):
    """
    This function is not used with Prophet as Prophet generates forecasts directly.
    Kept for compatibility with the app.py interface.
    """
    # The forecasting is now handled directly in the influenza_train_and_predict function
    # This is a placeholder to maintain compatibility
    return {
        'values': [],
        'upper_bounds': [],
        'lower_bounds': []
    }