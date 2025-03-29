import streamlit as st
from PIL import Image
from model import fetch_data, influenza_train_and_predict
from model import smape
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from constants import STATE_CODE_MAPPER
import threading
import time
import requests
from datetime import datetime

class KeepAlive:
    def __init__(self):
        self.running = False
        self.thread = None

    def keep_alive(self):
        while self.running:
            try:
                # Make request to your app URL
                requests.get("https://evision-python-app-gp4lssjsvbwz53dlnsem6m.streamlit.app/")
            except:
                pass
            time.sleep(60 * 10)  # Sleep for 10 minutes

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.keep_alive)
            self.thread.daemon = True
            self.thread.start()

# Initialize keep-alive in session state
if 'keep_alive' not in st.session_state:
    st.session_state.keep_alive = KeepAlive()
    st.session_state.keep_alive.start()

# Initialize session state for cache
if 'previous_predictions' not in st.session_state:
    st.session_state.previous_predictions = {}

INFLUENZA = "Influenza"

st.set_page_config(
    page_title="eVision",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

placeholder = st.empty()
predict = False
# if not predict:
if not predict:
    with placeholder.container():
        st.markdown(
            "<h1 style='text-align: center; color: grey;'>Psst! Hit Predict</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h2 style='text-align: center; color: black;'>To see the model in action</h2>",
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        model_image = Image.open("model.png")
        with col1:
            st.write(" ")

        with col2:
            st.image(model_image)

        with col3:
            st.write(" ")


with st.sidebar:
    scu = Image.open("scu-icon.png")
    epiclab = Image.open("EpicLab-icon.png")
    cepheid = Image.open("cepheid.png")
    
    scu_height = scu.height
    epiclab_height = epiclab.height
    cepheid_height = cepheid.height
    
    max_height = max(scu_height, epiclab_height)
    padding_needed = (max_height - cepheid_height) // 2
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(scu, use_container_width=True)
        with col2:
            for _ in range(padding_needed // 10):
                st.write("")
            st.image(cepheid, use_container_width=True)
        with col3:
            st.image(epiclab, use_container_width=True)

            
    st.header("eVision")
    st.write("**Using Prophet for Predictions**")
    disease = st.selectbox(
        "**Pick disease**",
        [INFLUENZA],
        help="This input is where you select the disease that you would like to make a prediction for. The only option we currently have is influenza, but more diseases can be easily added to this framework if you would like to expand the scope of these predictions.",
    )
    if disease == INFLUENZA:
        pred_level = st.selectbox(
            "**Prediction Level**",
            ["National", "State"],
            help="This input allows you to select the level of the area you would like to predict. The options in this case are national and state, since these are the two levels that the WHO and CDC are able to provide us with. There has been work done by our team to add the ability to predict for cities since we have the Google data available for it, but there is no case data out there for us to reference for predictions.",
        )
        if pred_level == "National":
            states = None
        elif pred_level == "State":
            state_list = STATE_CODE_MAPPER.keys()
            states = st.selectbox("**Pick State**", state_list)
        terms = st.multiselect(
            "**Keywords**",
            ["cough", "flu", "tamiflu", "sore throat"],
            help="This is where you type in the keywords you would like us to extract from Google Trends",
        )
        predict = False
        with st.form(disease + "_train"):
            num_weeks = st.select_slider(
                "**Number of weeks prediction**",
                [3, 7, 14],
                help="This input allows you to dictate how many weeks ahead you would like to predict cases for. Generally, the higher amount of weeks you select, the less accurate your predictions become.",
            )
            epochs = st.slider(
                "**Number of Epochs**",
                min_value=1,
                max_value=200,
                step=1,
                value=100,  # Default value for Prophet
                help="This input controls the model fit iterations. With Prophet, this affects the smoothness of the trend.",
            )
            predict = st.form_submit_button("**Predict**")
            if predict:
                placeholder.empty()

# Create a cache key for data and model predictions
def create_cache_key(terms, pred_level, states, epochs=None, num_weeks=None):
    base_key = f"{'-'.join(sorted(terms))}__{pred_level}__{states}"
    if epochs is not None and num_weeks is not None:
        return f"{base_key}__{epochs}__{num_weeks}"
    return base_key

# Data fetching phase - using st.cache_data decorator in model.py
df = None
if disease == INFLUENZA and terms:
    # Check if we have all the required values before fetching data
    with st.spinner("Fetching the data..."):
        df = fetch_data(terms, pred_level, states)
        # The fetch_data function is now cached with @st.cache_data in model.py

# Model prediction phase - using both decorator caching and session state caching
response = None
if disease == INFLUENZA and predict and df is not None:
    # Create a unique cache key for this prediction
    cache_key = create_cache_key(terms, pred_level, states, epochs, num_weeks)
    
    # Check if we have this prediction cached in session state
    if cache_key in st.session_state.previous_predictions:
        response = st.session_state.previous_predictions[cache_key]
        st.success("Using cached prediction results")
    else:
        # If not in cache, use the cached function (which may still return quickly if parameters match)
        with st.spinner("Training the Prophet model and making predictions..."):
            response = influenza_train_and_predict(df, epochs, num_weeks)
            
        # Store in session state cache for future use
        st.session_state.previous_predictions[cache_key] = response

    if response:
        st.header(f"{disease} Prediction results (Prophet Model)")
        
        # Get dates from response
        dates = response.get("dates", [])
        
        # Create dataframe for historical predictions
        results_df = pd.DataFrame({
            "actual_data": response.get("actual_data"),
            "predictions": response.get("predictions"),
            "date": dates[-len(response.get("actual_data")):],  # Match dates with actual data length
        })
        
        results_df["week"] = range(1, len(results_df) + 1)
        ci = response.get("confidence_interval")
        
        # For Prophet, we can use the built-in confidence intervals
        results_df["predictions_upper"] = results_df["predictions"] * (1 + 0.01 * ci)
        results_df["predictions_lower"] = results_df["predictions"] * (1 - 0.01 * ci)
        
        # Create two columns for the graphs
        col1, col2 = st.columns(2)
        
        # For the Historical Prediction graph
        with col1:
            st.markdown("""
            <h3 style="color: #31708f; margin-top: 0;">Historical Prediction</h3>
            """, unsafe_allow_html=True)
            
            fig1 = go.Figure()
            
            fig1.add_trace(
                go.Scatter(name="Actual Data", x=results_df["date"], y=results_df["actual_data"],
                        mode="lines", line=dict(color="rgb(31, 119, 180)"))
            )
            
            fig1.add_trace(
                go.Scatter(
                    name="Predictions",
                    x=results_df["date"],
                    y=results_df["predictions"],
                    mode="lines",
                    line=dict(color="rgb(255, 127, 14)")
                )
            )
            
            fig1.add_trace(
                go.Scatter(
                    name="Upper Bound",
                    x=results_df["date"],
                    y=results_df["predictions_upper"],
                    mode="lines",
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            
            fig1.add_trace(
                go.Scatter(
                    name="Lower Bound",
                    x=results_df["date"],
                    y=results_df["predictions_lower"],
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode="lines",
                    fillcolor="rgba(68, 68, 68, 0.2)",
                    fill="tonexty",
                    showlegend=False,
                )
            )
            
            fig1.update_layout(
                xaxis={"title": "Date", "tickangle": 45},
                yaxis={"title": "ILI Cases"},
                height=400,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig1, use_container_width=True)

        # For the Future Forecast graph - now using Prophet's built-in forecasts
        with col2:
            st.markdown(f"""
            <h3 style="color: #2c7fb8; margin-top: 0;">Future Forecast ({num_weeks} weeks)</h3>
            """, unsafe_allow_html=True)
            
            # Get future forecast data from Prophet
            future_dates = response.get("future_dates", [])
            future_predictions = response.get("future_predictions", [])
            future_upper = response.get("future_predictions_upper", [])
            future_lower = response.get("future_predictions_lower", [])
            
            # Include the last actual data point for continuity
            last_date = pd.to_datetime(results_df["date"].iloc[-1]).strftime('%Y-%m-%d')
            last_actual_value = results_df["actual_data"].iloc[-1]
            
            forecast_dates = [last_date] + future_dates
            forecast_values = [last_actual_value] + list(future_predictions)
            upper_values = [last_actual_value * (1 + 0.01 * ci)] + list(future_upper)
            lower_values = [last_actual_value * (1 - 0.01 * ci)] + list(future_lower)
            
            fig2 = go.Figure()
            
            # Add the forecast line
            fig2.add_trace(
                go.Scatter(
                    name="Prophet Forecast",
                    x=forecast_dates,
                    y=forecast_values,
                    mode="lines",
                    line=dict(color="rgb(214, 39, 40)")
                )
            )
            
            # Add the last actual data point marker
            fig2.add_trace(
                go.Scatter(
                    name="Last Actual", 
                    x=[forecast_dates[0]], 
                    y=[last_actual_value],
                    mode="markers",
                    marker=dict(color="rgb(31, 119, 180)", size=10),
                )
            )
            
            # Add confidence intervals from Prophet
            fig2.add_trace(
                go.Scatter(
                    name="Upper Bound",
                    x=forecast_dates,
                    y=upper_values,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            
            fig2.add_trace(
                go.Scatter(
                    name="Lower Bound",
                    x=forecast_dates,
                    y=lower_values,
                    line=dict(width=0),
                    mode="lines",
                    fillcolor="rgba(214, 39, 40, 0.2)",
                    fill="tonexty",
                    showlegend=False,
                )
            )
            
            fig2.update_layout(
                xaxis={"title": "Date", "tickangle": 45},
                yaxis={"title": "Predicted ILI Cases"},
                height=400,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Metrics in two columns
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric(
                "**Prophet Confidence Interval**", f'{response.get("confidence_interval"):.5f}'
            )

        with metric_col2:
            # Calculate SMAPE using the function already defined in model.py
            smape_value = smape(results_df["actual_data"].values, results_df["predictions"].values)

            # Display SMAPE metric
            st.metric(
                "**Error (SMAPE)**", 
                f"{smape_value:.5f}", 
                help="Symmetric Mean Absolute Percentage Error"
            )

        # Components visualization from Prophet
        st.header("Prophet Model Components")
        st.write("Prophet decomposes the time series into trend, seasonality, and holidays components.")
        
        # Mock loss graph for compatibility with original app
        st.header("Model Optimization")
        history = response.get("history")
        
        df_loss = pd.DataFrame(
            {
                "loss": history.history["loss"],
            }
        )

        df_loss["epoch"] = range(1, epochs + 1)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                name="Optimization Path",
                x=df_loss["epoch"],
                y=df_loss["loss"],
                mode="lines",
                line=dict(color="rgb(214, 39, 40)")
            )
        )
        fig.update_layout(
            xaxis={"title": "Iteration"},
            yaxis={"title": "Model Parameter Scale"},
            title="Prophet Model Optimization Path",
            title_x=0.5,
            height=300
        )
        st.plotly_chart(fig, theme=None, use_container_width=True)
        
        # Add Prophet model explanation
        st.header("About Prophet Model")
        st.write("""
        Meta's Prophet is a forecasting procedure that works best with time series data that have strong seasonal effects and several seasons of historical data. It's robust to missing data, shifts in trend, and large outliers.
        
        **Key advantages over the previous LSTM model:**
        - Better handling of seasonality (weekly, monthly, yearly patterns)
        - Automatic detection of trend changes
        - Robust to missing data points
        - Built-in uncertainty intervals
        - More interpretable components
        """)

# Add a button to clear cache if needed
with st.sidebar:
    if st.button("Clear Cache"):
        # Clear all caches
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.previous_predictions = {}
        st.success("Cache cleared successfully!")