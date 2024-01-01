import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime as dt
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from sklearn.metrics import mean_absolute_error

# load twitter ohlc data
df = pd.read_csv('../data/TWITTER.csv')
df.describe()
# data frame summary / check for missing values
df.info()

df["Date"] = pd.to_datetime(df["Date"])

# This script creates a 3x2 grid of subplots using Plotly, each plotting a different
# column from the DataFrame 'df' against the 'Date' column. Set to a size of
# 1500x1000 pixels, titled "Twitter Data".
columns = df.columns[1:]
fig = make_subplots(rows=3, cols=2, subplot_titles=columns)
for row in range(1, 4):
    for col in range(1, 3):
        column = columns[row * col - 1]
        fig.add_trace(go.Scatter(x = df['Date'], y = df[column]), row=row, col=col)
fig.update_layout(height=1500, width=1000, title_text="Twitter Data", showlegend=False)
fig.show()

# This adds a year column which is aggregated from the Date column.
# Then creates a pie chart for the sum of volume data by year.
df['Year'] = df['Date'].dt.year
df_pie = df.groupby('Year')['Volume'].sum()
layout = {
    'title': 'Pie Chart for Sum of Volume Data against Each Year'
}
fig = go.Figure(data=[go.Pie(labels=df_pie.index, values = df_pie.values, textinfo='label')], layout=layout)
fig.update_layout(height=800, width=600)
fig.show()