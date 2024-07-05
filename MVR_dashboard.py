#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

st.title('Stock Prices Analysis and Prediction Dashboard')
st.subheader('MasterCard and Visa (2008-2024)')

# Data Loading and Cleaning
st.header('Data Loading and Cleaning')
data = pd.read_csv('C:/Users/satyendra maurya/OneDrive/Desktop/streamlit_dashboards/MVR.csv')
st.write(data.head())

# Check for missing values
st.write('Missing Values:')
st.write(data.isnull().sum())

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
st.write('Data Types:')
st.write(data.dtypes)

st.write(data.info())

# Descriptive statistics
st.header('Descriptive Statistics')
st.write(data.describe())

# Time Series Visualization
st.header('Time Series Visualization')
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close_M'], label='MasterCard Close')
plt.plot(data.index, data['Close_V'], label='Visa Close')
plt.title('Stock Prices of MasterCard and Visa')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(plt)

# Calculate and plot moving averages
data['MA_Close_M'] = data['Close_M'].rolling(window=30).mean()
data['MA_Close_V'] = data['Close_V'].rolling(window=30).mean()

plt.figure(figsize=(14, 7))
plt.plot(data['Close_M'], label='MasterCard Close Price')
plt.plot(data['MA_Close_M'], label='MasterCard 30-Day MA')
plt.plot(data['Close_V'], label='Visa Close Price')
plt.plot(data['MA_Close_V'], label='Visa 30-Day MA')
plt.title('Moving Averages of Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt)

# Plot the volume of stocks traded
plt.figure(figsize=(14, 7))
plt.plot(data['Volume_M'], label='MasterCard Volume')
plt.plot(data['Volume_V'], label='Visa Volume')
plt.title('Volume of Stocks Traded')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
st.pyplot(plt)

# Calculate 50-day and 200-day Moving Averages for MasterCard and Visa
data['SMA50_M'] = data['Close_M'].rolling(window=50).mean()
data['SMA200_M'] = data['Close_M'].rolling(window=200).mean()
data['SMA50_V'] = data['Close_V'].rolling(window=50).mean()
data['SMA200_V'] = data['Close_V'].rolling(window=200).mean()

# Plot the moving averages along with the stock prices
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close_M'], label='MasterCard Close')
plt.plot(data.index, data['SMA50_M'], label='MasterCard SMA50')
plt.plot(data.index, data['SMA200_M'], label='MasterCard SMA200')
plt.title('MasterCard Stock Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(plt)

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close_V'], label='Visa Close')
plt.plot(data.index, data['SMA50_V'], label='Visa SMA50')
plt.plot(data.index, data['SMA200_V'], label='Visa SMA200')
plt.title('Visa Stock Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(plt)

# Volatility Analysis
st.header('Volatility Analysis')
data['Volatility_M'] = data['Close_M'].rolling(window=30).std()
data['Volatility_V'] = data['Close_V'].rolling(window=30).std()

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Volatility_M'], label='MasterCard Volatility')
plt.plot(data.index, data['Volatility_V'], label='Visa Volatility')
plt.title('Stock Price Volatility of MasterCard and Visa')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
st.pyplot(plt)

# Comparative Analysis
st.header('Comparative Analysis')
data['Return_M'] = data['Close_M'].pct_change()
data['Return_V'] = data['Close_V'].pct_change()
data['Cumulative_Return_M'] = (1 + data['Return_M']).cumprod()
data['Cumulative_Return_V'] = (1 + data['Return_V']).cumprod()

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Cumulative_Return_M'], label='MasterCard Cumulative Return')
plt.plot(data.index, data['Cumulative_Return_V'], label='Visa Cumulative Return')
plt.title('Cumulative Returns of MasterCard and Visa')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
st.pyplot(plt)

# Correlation Analysis
st.header('Correlation Analysis')
correlation = data[['Close_M', 'Close_V']].corr()
st.write(correlation)

# Seasonal Decomposition
st.header('Seasonal Decomposition')
decomposition_M = seasonal_decompose(data['Close_M'], model='multiplicative', period=365)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
ax1.plot(decomposition_M.observed)
ax1.set_title('Observed - MasterCard')
ax2.plot(decomposition_M.trend)
ax2.set_title('Trend - MasterCard')
ax3.plot(decomposition_M.seasonal)
ax3.set_title('Seasonal - MasterCard')
ax4.plot(decomposition_M.resid)
ax4.set_title('Residual - MasterCard')
plt.tight_layout()
st.pyplot(fig)

decomposition_V = seasonal_decompose(data['Close_V'], model='multiplicative', period=365)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
ax1.plot(decomposition_V.observed)
ax1.set_title('Observed - Visa')
ax2.plot(decomposition_V.trend)
ax2.set_title('Trend - Visa')
ax3.plot(decomposition_V.seasonal)
ax3.set_title('Seasonal - Visa')
ax4.plot(decomposition_V.resid)
ax4.set_title('Residual - Visa')
plt.tight_layout()
st.pyplot(fig)

# Augmented Dickey-Fuller Test
st.header('Stationarity Test (ADF Test)')
def adf_test(series):
    result = adfuller(series.dropna())
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])
    for key, value in result[4].items():
        st.write(f'Critical Value ({key}): {value}')

st.write("ADF Test for MasterCard Close Price:")
adf_test(data['Close_M'])

st.write("\nADF Test for Visa Close Price:")
adf_test(data['Close_V'])

# Machine Learning Predictions
st.header('Machine Learning Predictions')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_M = scaler.fit_transform(data[['Close_M']])
scaled_data_V = scaler.fit_transform(data[['Close_V']])

train_len_M = int(len(scaled_data_M) * 0.8)
train_len_V = int(len(scaled_data_V) * 0.8)

train_data_M = scaled_data_M[:train_len_M]
test_data_M = scaled_data_M[train_len_M:]

train_data_V = scaled_data_V[:train_len_V]
test_data_V = scaled_data_V[train_len_V:]

def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

seq_length = 60
x_train_M, y_train_M = create_sequences(train_data_M, seq_length)
x_test_M, y_test_M = create_sequences(test_data_M, seq_length)

x_train_V, y_train_V = create_sequences(train_data_V, seq_length)
x_test_V, y_test_V = create_sequences(test_data_V, seq_length)

x_train_M = np.reshape(x_train_M, (x_train_M.shape[0], x_train_M.shape[1], 1))
x_test_M = np.reshape(x_test_M, (x_test_M.shape[0], x_test_M.shape[1], 1))

x_train_V = np.reshape(x_train_V, (x_train_V.shape[0], x_train_V.shape[1], 1))
x_test_V = np.reshape(x_test_V, (x_test_V.shape[0], x_test_V.shape[1], 1))

# Building and training the LSTM model for MasterCard
st.subheader('LSTM Model - MasterCard')
model_M = Sequential()
model_M.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model_M.add(LSTM(50, return_sequences=False))
model_M.add(Dense(25))
model_M.add(Dense(1))
model_M.compile(optimizer='adam', loss='mean_squared_error')
model_M.fit(x_train_M, y_train_M, batch_size=1, epochs=1)

# Predictions for MasterCard
predictions_M = model_M.predict(x_test_M)
predictions_M = scaler.inverse_transform(predictions_M)
rmse_M = np.sqrt(mean_squared_error(y_test_M, predictions_M))
st.write(f'MasterCard LSTM RMSE: {rmse_M}')

# Plot the predictions for MasterCard
train_M = data[:train_len_M]['Close_M']
valid_M = data[train_len_M:]['Close_M'].to_frame()
valid_M = valid_M.iloc[:len(predictions_M)]
valid_M['Predictions'] = predictions_M

plt.figure(figsize=(14, 7))
plt.plot(train_M, label='Train - MasterCard')
plt.plot(valid_M['Close_M'], label='Valid - MasterCard')
plt.plot(valid_M['Predictions'], label='Predictions - MasterCard')
plt.title('LSTM Model - MasterCard')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend(['Train', 'Valid', 'Predictions'], loc='lower right')
st.pyplot(plt)


# Building and training the LSTM model for Visa
st.subheader('LSTM Model - Visa')
model_V = Sequential()
model_V.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model_V.add(LSTM(50, return_sequences=False))
model_V.add(Dense(25))
model_V.add(Dense(1))
model_V.compile(optimizer='adam', loss='mean_squared_error')
model_V.fit(x_train_V, y_train_V, batch_size=1, epochs=1)

# Predictions for Visa
predictions_V = model_V.predict(x_test_V)
predictions_V = scaler.inverse_transform(predictions_V)
rmse_V = np.sqrt(mean_squared_error(y_test_V, predictions_V))
st.write(f'Visa LSTM RMSE: {rmse_V}')

# Plot the predictions for Visa
train_V = data[:train_len_V]['Close_V']
valid_V = data[train_len_V:]['Close_V'].to_frame()
valid_V = valid_V.iloc[:len(predictions_V)]
valid_V['Predictions'] = predictions_V

plt.figure(figsize=(14, 7))
plt.plot(train_V, label='Train - Visa')
plt.plot(valid_V['Close_V'], label='Valid - Visa')
plt.plot(valid_V['Predictions'], label='Predictions - Visa')
plt.title('LSTM Model - Visa')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend(['Train', 'Valid', 'Predictions'], loc='lower right')
st.pyplot(plt)


# ARIMA Model
st.subheader('ARIMA Model - MasterCard and Visa')
def arima_model(data, order):
    model = ARIMA(data, order=order)
    fitted = model.fit()
    return fitted

# Predicting with ARIMA
arima_order = (5, 1, 0)
arima_M = arima_model(data['Close_M'], arima_order)
arima_pred_M = arima_M.predict(start=train_len_M, end=len(data)-1, typ='levels')

arima_V = arima_model(data['Close_V'], arima_order)
arima_pred_V = arima_V.predict(start=train_len_V, end=len(data)-1, typ='levels')

# Plotting ARIMA predictions
plt.figure(figsize=(14, 7))
plt.plot(data['Close_M'], label='MasterCard Actual')
plt.plot(arima_pred_M, label='MasterCard ARIMA Prediction')
plt.title('ARIMA Model - MasterCard')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
st.pyplot(plt)

plt.figure(figsize=(14, 7))
plt.plot(data['Close_V'], label='Visa Actual')
plt.plot(arima_pred_V, label='Visa ARIMA Prediction')
plt.title('ARIMA Model - Visa')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
st.pyplot(plt)

# Add a section for ARIMA predictions and candlestick chart
st.subheader('Future Stock Price Predictions and Candlestick Chart')

# Function to extend predictions using ARIMA
def predict_stock_price(data, column_name, forecast_periods):
    train_size = int(len(data) * 0.8)
    train, test = data[column_name][:train_size], data[column_name][train_size:]

    # Fit the ARIMA model
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast future periods
    future_dates = pd.date_range(start=data.index[-1], periods=forecast_periods, freq='B')
    forecast = model_fit.forecast(steps=forecast_periods)
    forecast_series = pd.Series(forecast, index=future_dates)
    
    return forecast_series

# Predict stock prices and create the candlestick chart
forecast_periods = 3 * 252  # Approximately 252 trading days per year
forecast_M = predict_stock_price(data, 'Close_M', forecast_periods)
forecast_V = predict_stock_price(data, 'Close_V', forecast_periods)

# Combine historical and forecast data
extended_data_M = pd.concat([data['Close_M'], forecast_M])
extended_data_V = pd.concat([data['Close_V'], forecast_V])

# Create a DataFrame for candlestick chart
candlestick_data_M = pd.DataFrame({
    'Date': extended_data_M.index,
    'Open': extended_data_M.shift(1).fillna(method='bfill'),
    'High': extended_data_M.rolling(2).max(),
    'Low': extended_data_M.rolling(2).min(),
    'Close': extended_data_M
}).reset_index(drop=True)

candlestick_data_V = pd.DataFrame({
    'Date': extended_data_V.index,
    'Open': extended_data_V.shift(1).fillna(method='bfill'),
    'High': extended_data_V.rolling(2).max(),
    'Low': extended_data_V.rolling(2).min(),
    'Close': extended_data_V
}).reset_index(drop=True)

# Plot the candlestick chart using Plotly
fig = go.Figure()

# MasterCard Candlestick
fig.add_trace(go.Candlestick(
    x=candlestick_data_M['Date'],
    open=candlestick_data_M['Open'],
    high=candlestick_data_M['High'],
    low=candlestick_data_M['Low'],
    close=candlestick_data_M['Close'],
    name='MasterCard',
    increasing_line_color='blue', decreasing_line_color='red'
))

# Visa Candlestick
fig.add_trace(go.Candlestick(
    x=candlestick_data_V['Date'],
    open=candlestick_data_V['Open'],
    high=candlestick_data_V['High'],
    low=candlestick_data_V['Low'],
    close=candlestick_data_V['Close'],
    name='Visa',
    increasing_line_color='green', decreasing_line_color='orange'
))

# Update layout
fig.update_layout(
    title='MasterCard and Visa Stock Prices (Historical and Predicted)',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False
)

# Display the candlestick chart in Streamlit
st.plotly_chart(fig)
