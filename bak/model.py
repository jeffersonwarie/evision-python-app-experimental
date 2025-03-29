import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import pandas as pd
from app_logger import setup_logger
from typing import Dict, Any, List
import streamlit as st
import requests
from io import StringIO

logging = setup_logger(__name__)

# Base URL for the data files
BASE_URL = "http://epidemicvision.com/pydata"

# Define the CSV file URLs
TRENDS_NATIONAL_CSV = f"{BASE_URL}/google_trends-National.csv"
TRENDS_STATE_CSV = f"{BASE_URL}/google_trends-State.csv"
ILINET_NATIONAL_CSV = f"{BASE_URL}/ILINet-National.csv"
ILINET_STATE_CSV = f"{BASE_URL}/ILINet-State.csv"

def smape(a, f):
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
    # Create a copy of data to avoid modifying the cached dataframe
    data = data.copy()
    
    dates = data["date"].to_list()
    data.drop(columns=["date"], inplace=True)
    data = data.astype(float)
    pred_col = "ilitotal"
    batch_size = 256
    X, y = data.iloc[:-predict_ahead_by, :].copy(deep=True), data.iloc[
        predict_ahead_by:, :
    ].loc[:, [pred_col]].copy(deep=True)

    std = y[pred_col].std(ddof=0)
    mean = y[pred_col].mean()
    y[pred_col] = (y[pred_col] - mean) / std
    X[pred_col] = (X[pred_col] - mean) / std

    train_test_split = 0.75
    trainX, testX = (
        X[: round(X.shape[0] * train_test_split)].to_numpy(),
        X[round(X.shape[0] * train_test_split) :].to_numpy(),
    )
    trainy, testy = (
        y[: round(y.shape[0] * train_test_split)].to_numpy(),
        y[round(y.shape[0] * train_test_split) :].to_numpy(),
    )
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    trainy = np.reshape(trainy, (trainy.shape[0], trainy.shape[1], 1))
    testy = np.reshape(testy, (testy.shape[0], testy.shape[1], 1))

    model1 = Sequential()
    model1.add(LSTM(327, input_shape=(trainX.shape[1], trainX.shape[2])))
    model1.add(Dropout(rate=0.1))
    model1.add(Dense(1))

    model1.compile(loss="mse", optimizer="adam")
    logging.info(f"Finished compiling the model. Starting training")

    response = {}
    response["dates"] = dates

    history = model1.fit(
        trainX, trainy, batch_size=batch_size, epochs=epochs, shuffle=False
    )
    response["history"] = history
    pred = model1.predict(testX)
    pred_unnorm = (pred * std) + mean
    testy_unnorm = (testy * std) + mean

    response["predictions"] = pred_unnorm.reshape(testy.shape[0])
    response["actual_data"] = testy_unnorm.reshape(testy.shape[0])
    ci = smape(
        pred_unnorm.reshape(testy.shape[0]), testy_unnorm.reshape(testy.shape[0])
    )
    response["confidence_interval"] = ci

    return response

def generate_future_forecast(model, last_input_data, num_weeks, confidence_interval):
    """
    Generate future forecast using the trained model.
    
    Args:
        model: Trained Keras model
        last_input_data: The last input data used for prediction
        num_weeks: Number of weeks to forecast
        confidence_interval: Confidence interval value for error bounds
        
    Returns:
        Dictionary containing forecast values and confidence intervals
    """
    import numpy as np
    
    # Initialize storage for forecasts
    forecasts = []
    
    # Start with the last input
    current_input = last_input_data.copy()
    
    # Reshape for prediction
    input_shape = (1, current_input.shape[1], 1)
    
    # For each future week, generate the forecast
    for i in range(num_weeks):
        # Reshape input for prediction
        input_reshaped = np.reshape(current_input, input_shape)
        
        # Generate prediction
        pred = model.predict(input_reshaped, verbose=0)
        forecasts.append(float(pred[0][0]))
        
        # Update input for next prediction by shifting window
        # This is a simplified approach - in a real system you'd need to update all features
        current_input = np.roll(current_input, -1)
        current_input[0, -1] = forecasts[-1]
    
    # Calculate confidence intervals
    ci_factor = confidence_interval / 100
    upper_bounds = [val * (1 + ci_factor) for val in forecasts]
    lower_bounds = [val * (1 - ci_factor) for val in forecasts]
    
    return {
        'values': forecasts,
        'upper_bounds': upper_bounds,
        'lower_bounds': lower_bounds
    }
