import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime as dt
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from sklearn.metrics import mean_absolute_error

# Load Twitter Data and create DF change datatype for date to datetime
df = pd.read_csv('../data/TWITTER.csv')
df["Date"] = pd.to_datetime(df["Date"])

# Create new DF for prophet prediction model
df2 = df[["Date", "Close"]]
df2.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

# Load Prophet and fit to our twitter forecasting DF
m = Prophet()
m.fit(df2)

# Create Future Dates
future_prices = m.make_future_dataframe(periods=365, freq='D')

# Predict Future Prices
forecast = m.predict(future_prices)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# 1 Year Forecast and creating subplots
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
plt.title("Daily Prediction \n 1 Year Forecast")
plt.show()

# Line Chart w/ the 3 subplots
fig = m.plot_components(forecast)
ax = fig.gca()
plt.show()