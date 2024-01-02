import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime as dt
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('../data/TWITTER.csv')
df["Date"] = pd.to_datetime(df["Date"])

df2 = df[["Date", "Close"]]
df2.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

m = Prophet()
m.fit(df2)

future_prices = m.make_future_dataframe(periods=365, freq='D')

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

# Monthly Data Predictions
m = Prophet(changepoint_prior_scale=0.03).fit(df2)
future = m.make_future_dataframe(periods=12, freq='M')
fcst = m.predict(future)

fig = m.plot(fcst)
a = add_changepoints_to_plot(fig.gca(), m, fcst)
plt.title("Monthly Prediction \n 1 year time frame")
plt.show()

fig = m.plot_components(fcst)
ax = fig.gca()
plt.show()

## Evaluate Model
y_true = df2['y'].values
y_pred = fcst['yhat'][:-12].values
mae = mean_absolute_error(y_true, y_pred)
mae

# Graph for Visual Analysis
fig = go.Figure()
fig.add_trace(go.Scatter(x=df2['ds'], y=y_true, mode='lines',name="Actual"))
fig.add_trace(go.Scatter(x=df2['ds'], y=y_pred, mode='lines',name='Predicted'))
fig['layout'].update(title="Line Chart for Actual and Predicted Values")
fig.show()
