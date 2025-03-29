import numpy as np
import pandas as pd
from app_logger import setup_logger
from typing import Dict, Any, List
import streamlit as st
import requests
from io import StringIO
from tqdm import tqdm

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

##### Activation Functions #####
def sigmoid(input, derivative=False):
    if derivative:
        return input * (1 - input)
    
    return 1 / (1 + np.exp(-input))

def tanh(input, derivative=False):
    if derivative:
        return 1 - input ** 2
    
    return np.tanh(input)

def softmax(input):
    exp_vals = np.exp(input - np.max(input, axis=0, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=0, keepdims=True)

##### Helper Functions #####
# Xavier Normalized Initialization
def initWeights(input_size, output_size):
    return np.random.uniform(-1, 1, (output_size, input_size)) * np.sqrt(6 / (input_size + output_size))

##### Long Short-Term Memory Network Class #####
class LSTM:
    def __init__(self, input_size, hidden_size, output_size, num_epochs, learning_rate):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.input_size = input_size
        self.output_size = output_size

        # Forget Gate
        self.wf = initWeights(input_size, hidden_size)
        self.bf = np.zeros((hidden_size, 1))

        # Input Gate
        self.wi = initWeights(input_size, hidden_size)
        self.bi = np.zeros((hidden_size, 1))

        # Candidate Gate
        self.wc = initWeights(input_size, hidden_size)
        self.bc = np.zeros((hidden_size, 1))

        # Output Gate
        self.wo = initWeights(input_size, hidden_size)
        self.bo = np.zeros((hidden_size, 1))

        # Final Gate
        self.wy = initWeights(hidden_size, output_size)
        self.by = np.zeros((output_size, 1))
        
        # Store loss history
        self.loss_history = []

    # Reset Network Memory
    def reset(self):
        self.concat_inputs = {}

        self.hidden_states = {-1: np.zeros((self.hidden_size, 1))}
        self.cell_states = {-1: np.zeros((self.hidden_size, 1))}

        self.activation_outputs = {}
        self.candidate_gates = {}
        self.output_gates = {}
        self.forget_gates = {}
        self.input_gates = {}
        self.outputs = {}

    # Forward Propagation
    def forward(self, inputs):
        self.reset()

        outputs = []
        for q in range(len(inputs)):
            self.concat_inputs[q] = np.concatenate((self.hidden_states[q - 1], inputs[q]))

            self.forget_gates[q] = sigmoid(np.dot(self.wf, self.concat_inputs[q]) + self.bf)
            self.input_gates[q] = sigmoid(np.dot(self.wi, self.concat_inputs[q]) + self.bi)
            self.candidate_gates[q] = tanh(np.dot(self.wc, self.concat_inputs[q]) + self.bc)
            self.output_gates[q] = sigmoid(np.dot(self.wo, self.concat_inputs[q]) + self.bo)

            self.cell_states[q] = self.forget_gates[q] * self.cell_states[q - 1] + self.input_gates[q] * self.candidate_gates[q]
            self.hidden_states[q] = self.output_gates[q] * tanh(self.cell_states[q])

            outputs += [np.dot(self.wy, self.hidden_states[q]) + self.by]

        return outputs

    # Backward Propagation
    def backward(self, errors, inputs):
        d_wf, d_bf = 0, 0
        d_wi, d_bi = 0, 0
        d_wc, d_bc = 0, 0
        d_wo, d_bo = 0, 0
        d_wy, d_by = 0, 0

        dh_next, dc_next = np.zeros_like(self.hidden_states[0]), np.zeros_like(self.cell_states[0])
        for q in reversed(range(len(inputs))):
            error = errors[q]

            # Final Gate Weights and Biases Errors
            d_wy += np.dot(error, self.hidden_states[q].T)
            d_by += error

            # Hidden State Error
            d_hs = np.dot(self.wy.T, error) + dh_next

            # Output Gate Weights and Biases Errors
            d_o = tanh(self.cell_states[q]) * d_hs * sigmoid(self.output_gates[q], derivative=True)
            d_wo += np.dot(d_o, self.concat_inputs[q].T)
            d_bo += d_o

            # Cell State Error
            d_cs = tanh(tanh(self.cell_states[q]), derivative=True) * self.output_gates[q] * d_hs + dc_next

            # Forget Gate Weights and Biases Errors
            d_f = d_cs * self.cell_states[q - 1] * sigmoid(self.forget_gates[q], derivative=True)
            d_wf += np.dot(d_f, self.concat_inputs[q].T)
            d_bf += d_f

            # Input Gate Weights and Biases Errors
            d_i = d_cs * self.candidate_gates[q] * sigmoid(self.input_gates[q], derivative=True)
            d_wi += np.dot(d_i, self.concat_inputs[q].T)
            d_bi += d_i
            
            # Candidate Gate Weights and Biases Errors
            d_c = d_cs * self.input_gates[q] * tanh(self.candidate_gates[q], derivative=True)
            d_wc += np.dot(d_c, self.concat_inputs[q].T)
            d_bc += d_c

            # Concatenated Input Error (Sum of Error at Each Gate!)
            d_z = np.dot(self.wf.T, d_f) + np.dot(self.wi.T, d_i) + np.dot(self.wc.T, d_c) + np.dot(self.wo.T, d_o)

            # Error of Hidden State and Cell State at Next Time Step
            dh_next = d_z[:self.hidden_size, :]
            dc_next = self.forget_gates[q] * d_cs

        for d_ in (d_wf, d_bf, d_wi, d_bi, d_wc, d_bc, d_wo, d_bo, d_wy, d_by):
            np.clip(d_, -1, 1, out=d_)

        self.wf += d_wf * self.learning_rate
        self.bf += d_bf * self.learning_rate

        self.wi += d_wi * self.learning_rate
        self.bi += d_bi * self.learning_rate

        self.wc += d_wc * self.learning_rate
        self.bc += d_bc * self.learning_rate

        self.wo += d_wo * self.learning_rate
        self.bo += d_bo * self.learning_rate

        self.wy += d_wy * self.learning_rate
        self.by += d_by * self.learning_rate

    # Train method for numerical data (adapted for time series)
    def train_numerical(self, X_train, y_train, batch_size=32):
        n_samples = X_train.shape[0]
        
        # Initialize X as 3D array: [samples, timesteps=1, features]
        X_train_reshaped = X_train.reshape(n_samples, 1, -1)
        
        losses = []
        for epoch in tqdm(range(self.num_epochs)):
            epoch_loss = 0
            # Process mini-batches
            for i in range(0, n_samples, batch_size):
                batch_X = X_train_reshaped[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                batch_loss = 0
                for j in range(len(batch_X)):
                    # Convert to column vectors
                    x = batch_X[j].T
                    target = batch_y[j].reshape(-1, 1)
                    
                    # Forward pass
                    pred = self.forward(x)[0]
                    
                    # Calculate MSE loss
                    error = pred - target
                    batch_loss += np.mean(error ** 2)
                    
                    # Backward pass
                    self.backward([error], self.concat_inputs)
                
                epoch_loss += batch_loss / len(batch_X)
            
            avg_loss = epoch_loss / (n_samples // batch_size)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}, Loss: {avg_loss}")
        
        self.loss_history = losses
        return losses
    
    # Predict method for numerical data
    def predict(self, X):
        n_samples = X.shape[0]
        X_reshaped = X.reshape(n_samples, 1, -1)
        
        predictions = []
        for i in range(n_samples):
            x = X_reshaped[i].T
            pred = self.forward(x)[0]
            predictions.append(pred.flatten())
            
        return np.array(predictions)

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

def influenza_train_and_predict(
    data: pd.DataFrame, epochs: int, predict_ahead_by: int
) -> Dict[str, Any]:
    dates = data["date"].to_list()
    data.drop(columns=["date"], inplace=True)
    data = data.astype(float)
    pred_col = "ilitotal"
    
    # Split data
    X, y = data.iloc[:-predict_ahead_by, :].copy(deep=True), data.iloc[
        predict_ahead_by:, :
    ].loc[:, [pred_col]].copy(deep=True)

    # Get statistics for normalization
    std = y[pred_col].std(ddof=0)
    mean = y[pred_col].mean()
    
    # Normalize data
    X_scaled = X.copy()
    X_scaled[pred_col] = (X_scaled[pred_col] - mean) / std
    
    y_scaled = y.copy()
    y_scaled[pred_col] = (y_scaled[pred_col] - mean) / std

    # Train-test split
    train_test_split = 0.75
    train_idx = round(X_scaled.shape[0] * train_test_split)
    
    trainX = X_scaled[:train_idx].values
    testX = X_scaled[train_idx:].values
    
    trainy = y_scaled[:train_idx][pred_col].values.reshape(-1, 1)
    testy = y_scaled[train_idx:][pred_col].values.reshape(-1, 1)
    
    # Create custom LSTM model
    input_size = trainX.shape[1]
    hidden_size = 327  # Match original model
    output_size = 1    # Single output (ilitotal prediction)
    
    model = LSTM(
        input_size=input_size + hidden_size,  # Input features + hidden state
        hidden_size=hidden_size,
        output_size=output_size,
        num_epochs=epochs,
        learning_rate=0.01  # Adjust as needed
    )
    
    logging.info(f"Finished compiling the model. Starting training")
    
    # Train the model using our custom method for numerical data
    history = model.train_numerical(trainX, trainy, batch_size=256)
    
    # Make predictions
    pred = model.predict(testX)
    
    # Unnormalize predictions and actual values
    pred_unnorm = (pred * std) + mean
    testy_unnorm = (testy * std) + mean
    
    # Create history object similar to Keras format
    history_obj = {
        'epoch': list(range(len(model.loss_history))),
        'history': {'loss': model.loss_history}
    }
    
    # Prepare response
    response = {}
    response["dates"] = dates
    response["history"] = history_obj
    response["predictions"] = pred_unnorm.flatten()
    response["actual_data"] = testy_unnorm.flatten()
    ci = smape(pred_unnorm.flatten(), testy_unnorm.flatten())
    response["confidence_interval"] = ci
    
    return response