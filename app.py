import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from prophet import Prophet
from tensorflow.keras.models import load_model

# Function to load the best model for a given currency
def load_best_model(currency):
    models_dir = 'models'
    # Clean the currency name to match the saved filename
    safe_currency_name = currency.replace(' ', '_').replace('/', '_')

    # Try loading LSTM model first
    lstm_model_path = os.path.join(models_dir, f"{safe_currency_name}_lstm_model.h5")
    lstm_scaler_path = os.path.join(models_dir, f"{safe_currency_name}_lstm_scaler.pkl")
    if os.path.exists(lstm_model_path) and os.path.exists(lstm_scaler_path):
        try:
            model = load_model(lstm_model_path)
            scaler = joblib.load(lstm_scaler_path)
            st.write(f"Loaded LSTM model for {currency}")
            return model, scaler, 'LSTM'
        except Exception as e:
            st.error(f"Error loading LSTM model for {currency}: {e}")


    # Try loading LightGBM model
    lightgbm_model_path = os.path.join(models_dir, f"{safe_currency_name}_lightgbm_model.pkl")
    if os.path.exists(lightgbm_model_path):
         try:
            model = joblib.load(lightgbm_model_path)
            st.write(f"Loaded LightGBM model for {currency}")
            return model, None, 'LightGBM' # LightGBM doesn't use a separate scaler in this implementation
         except Exception as e:
            st.error(f"Error loading LightGBM model for {currency}: {e}")

    # Try loading XGBoost model
    xgboost_model_path = os.path.join(models_dir, f"{safe_currency_name}_xgboost_model.pkl")
    if os.path.exists(xgboost_model_path):
        try:
            model = joblib.load(xgboost_model_path)
            st.write(f"Loaded XGBoost model for {currency}")
            return model, None, 'XGBoost' # XGBoost doesn't use a separate scaler
        except Exception as e:
            st.error(f"Error loading XGBoost model for {currency}: {e}")

    # Try loading Prophet model
    prophet_model_path = os.path.join(models_dir, f"{safe_currency_name}_prophet_model.pkl")
    if os.path.exists(prophet_model_path):
        try:
            model = joblib.load(prophet_model_path)
            st.write(f"Loaded Prophet model for {currency}")
            return model, None, 'Prophet' # Prophet doesn't use a separate scaler
        except Exception as e:
            st.error(f"Error loading Prophet model for {currency}: {e}")

    # Try loading ARIMA model
    arima_model_path = os.path.join(models_dir, f"{safe_currency_name}_arima_model.pkl")
    if os.path.exists(arima_model_path):
        try:
            model = joblib.load(arima_model_path)
            st.write(f"Loaded ARIMA model for {currency}")
            return model, None, 'ARIMA' # ARIMA doesn't use a separate scaler
        except Exception as e:
            st.error(f"Error loading ARIMA model for {currency}: {e}")

    # Handle AutoTS loading if implemented
    # autots_model_path = os.path.join(models_dir, f"{safe_currency_name}_autots_model.pkl")
    # if os.path.exists(autots_model_path):
    #     try:
    #         model = joblib.load(autots_model_path)
    #         st.write(f"Loaded AutoTS model for {currency}")
    #         return model, None, 'AutoTS'
    #     except Exception as e:
    #         st.error(f"Error loading AutoTS model for {currency}: {e}")


    st.error(f"No trained model found for {currency}")
    return None, None, None

# Function to make forecast based on the loaded model type
def make_forecast(model, scaler, model_type, historical_data, forecast_horizon):
    try:
        if model_type == 'Prophet':
            future = model.make_future_dataframe(periods=forecast_horizon, freq='D')
            forecast = model.predict(future)
            # Select only the forecast period
            forecast_data = forecast[['ds', 'yhat']].tail(forecast_horizon).rename(columns={'ds': 'Date', 'yhat': 'Forecast'})
            return forecast_data

        elif model_type == 'ARIMA':
            # ARIMA forecast from statsmodels returns a Series
            forecast = model.forecast(steps=forecast_horizon)
            # Create a date range for the forecast horizon starting from the day after the last historical date
            last_date = historical_data['Time Serie'].max()
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
            forecast_data = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast.values})
            return forecast_data


        elif model_type in ['XGBoost', 'LightGBM']:
            # For XGBoost and LightGBM, we need to create future dates and convert them to timestamps
            last_date = historical_data['Time Serie'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
            X_future = future_dates.to_series().apply(lambda x: x.timestamp()).values.reshape(-1, 1)
            forecast_values = model.predict(X_future)
            forecast_data = pd.DataFrame({'Date': future_dates, 'Forecast': forecast_values})
            return forecast_data

        elif model_type == 'LSTM':
            # LSTM forecasting requires sequential data. We need to use the last 'seq_length' data points
            # from the historical data to predict the first future point, then use a sliding window.
            seq_length = 10 # This should match the sequence length used during training

            if len(historical_data) < seq_length:
                st.warning(f"Historical data is too short for LSTM forecasting (requires at least {seq_length} data points).")
                return pd.DataFrame()

            # Get the last 'seq_length' data points and scale them
            last_sequence = historical_data.tail(seq_length)[historical_data.columns[-1]].values.reshape(-1, 1)
            scaled_last_sequence = scaler.transform(last_sequence)

            forecasted_values = []
            current_sequence = scaled_last_sequence.reshape(1, seq_length, 1)

            for _ in range(forecast_horizon):
                predicted_value_scaled = model.predict(current_sequence, verbose=0)[0][0]
                forecasted_values.append(predicted_value_scaled)

                # Update the sequence for the next prediction (sliding window)
                current_sequence = np.append(current_sequence[:, 1:, :], [[predicted_value_scaled]], axis=1)

            # Inverse transform the forecasted values
            forecasted_values = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1))

            # Create future dates
            last_date = historical_data['Time Serie'].max()
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

            forecast_data = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecasted_values.flatten()})
            return forecast_data

        # Handle AutoTS forecasting if implemented
        # elif model_type == 'AutoTS':
        #     # Assuming AutoTS forecast method exists and returns a dataframe
        #     forecast = model.predict() # Check AutoTS documentation for actual method
        #     # Need to extract and format the forecast data similar to Prophet or other models
        #     # Example: forecast_data = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Forecast'})
        #     st.warning("AutoTS forecasting not fully implemented in this app.")
        #     return pd.DataFrame()

        else:
            st.error(f"Forecasting not implemented for model type: {model_type}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error during forecasting: {e}")
        return pd.DataFrame()


# --- Streamlit App ---
st.title("Currency Exchange Rate Forecasting")

# Load the processed data (assuming df is available from previous steps or loaded here)
# For a standalone app, you would load the data here.
# Since this is in Colab, we assume df is in the environment or load it.
try:
    # Attempt to load the cleaned dataframe if it's not in the environment
    # This path should match where your cleaned data is saved or accessible
    # If running as a standalone app, you might load from the original Excel and re-process or load a saved cleaned df.
    # For demonstration in Colab, we'll assume 'df' is available from prior cells.
    # In a real app, you'd load and preprocess the data here or load a preprocessed version.
    # For now, let's assume 'df' exists in the Colab environment for simplicity.
    # Replace this with actual data loading and preprocessing if running standalone.
    if 'df' not in st.session_state:
        st.warning("Assuming 'df' DataFrame is available from previous steps in Colab.")
        # In a real app, load and process data:
        # df = pd.read_excel("/tmp/Foreign_Exchange_Rates.xlsx", header=None)
        # df = df[0].str.split(',', expand=True)
        # column_names = df.iloc[0].tolist()
        # df.columns = column_names
        # df = df[1:].reset_index(drop=True)
        # df['Time Serie'] = pd.to_datetime(df['Time Serie'], dayfirst=True, errors='coerce')
        # df.replace('ND', np.nan, inplace=True)
        # for col in df.columns[2:-1]:
        #      df[col] = pd.to_numeric(df[col], errors='coerce')
        # st.session_state['df'] = df # Store in session state

        # For this Colab example, just rely on the global 'df' if it exists
        # If running standalone, the above loading/processing is needed.
        pass # Relying on global df in Colab for this example

    if 'df' in st.session_state:
        historical_df = st.session_state['df']
    else:
         # Fallback if df is not in session_state (e.g., running app.py directly after cleanup)
         # In a real app, you MUST load and preprocess your data here.
         # For Colab demo, let's try to access the global df again or indicate missing data.
         if 'df' in globals():
             historical_df = globals()['df']
             st.warning("Using global 'df' variable. For standalone app, load data explicitly.")
         else:
             st.error("Historical data ('df' DataFrame) not found. Please run the data loading and preprocessing steps.")
             st.stop()


except Exception as e:
    st.error(f"Error loading or accessing historical data: {e}")
    st.stop()


# Get the list of currency columns (excluding the first two and the last one)
currency_columns = historical_df.columns[2:-1].tolist()

# Currency selection dropdown
selected_currency = st.selectbox("Select Currency:", currency_columns)

# Forecast horizon input
forecast_horizon = st.slider("Select Forecast Horizon (days):", min_value=1, max_value=365, value=30)

# Button to trigger forecast
if st.button("Generate Forecast"):
    if selected_currency:
        model, scaler, model_type = load_best_model(selected_currency)

        if model:
            st.subheader(f"Forecasting {selected_currency}")

            # Make the forecast
            historical_currency_data = historical_df[['Time Serie', selected_currency]].copy()
            forecast_data = make_forecast(model, scaler, model_type, historical_currency_data, forecast_horizon)

            if not forecast_data.empty:
                st.subheader("Forecast Table:")
                st.dataframe(forecast_data)

                st.subheader("Forecast Chart:")
                # Combine historical and forecast data for plotting
                historical_currency_data_plot = historical_currency_data.rename(columns={'Time Serie': 'Date', selected_currency: 'Actual'})
                # Ensure 'Actual' column is numeric for plotting, coercing errors
                historical_currency_data_plot['Actual'] = pd.to_numeric(historical_currency_data_plot['Actual'], errors='coerce')


                combined_data = pd.merge(historical_currency_data_plot, forecast_data, on='Date', how='outer')

                # Use Melt to reshape for Streamlit chart compatibility
                combined_data_melted = combined_data.melt(
                    'Date',
                    value_vars=['Actual', 'Forecast'],
                    var_name='Type',
                    value_name='Exchange Rate'
                )

                st.line_chart(combined_data_melted, x='Date', y='Exchange Rate', color='Type')


            else:
                st.warning("Forecast could not be generated.")
        else:
            st.warning("Could not load the best model for the selected currency.")
    else:
        st.warning("Please select a currency.")
