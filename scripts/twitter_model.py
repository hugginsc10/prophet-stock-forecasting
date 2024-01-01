import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime as dt


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

# OHLC Chart w/ notes
data = go.Ohlc(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                increasing=dict(line=dict(color= '#58FA58')),
                decreasing=dict(line=dict(color= '#FA5858')))

layout = {
    'title': 'Twitter Stocks',
    'xaxis': {'title': 'Date',
             'rangeslider': {'visible': False}},
    'yaxis': {'title': 'Stock Price (USD$)'},
    'shapes': [{
        'x0': '2015-10-05', 'x1': '2015-10-05',
        'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper',
        'line': {'color': 'rgb(30,30,30)', 'width': 1}
        },
        {
        'x0': '2020-03-15', 'x1': '2020-03-15',
        'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper',
        'line': {'color': 'rgb(30,30,30)', 'width': 1}
    }
    ],
    'annotations': [{
        'x': '2015-10-05', 'y': 0.6, 'xref': 'x', 'yref': 'paper',
        'showarrow': False, 'xanchor': 'left',
        'text': 'Jack Dorsey becomes CEO of Twitter.'
        },
        {
        'x': '2020-03-15', 'y': 0.05, 'xref': 'x', 'yref': 'paper',
        'showarrow': False, 'xanchor': 'left',
        'text': 'Lockdown started in USA due to Covid19.'
    }
    ]
}

fig = go.Figure(data=[data], layout=layout)
fig.update(layout_xaxis_rangeslider_visible=True)
fig.show()

# Candlestick Chart
after_covid = df.loc[df['Date'] > '2020-03-15']
max_value = after_covid.iloc[:,1:-2].max().max()

g = after_covid.groupby(["Date"])
monthly_averages = g.aggregate({"Open": np.mean, "High": np.mean, "Low": np.mean, "Close":np.mean})
monthly_averages.reset_index(level=0, inplace=True)

trace = go.Candlestick(x=monthly_averages['Date'],
                       open=monthly_averages['Open'].values.tolist(),
                       high=monthly_averages['High'].values.tolist(),
                       low=monthly_averages['Low'].values.tolist(),
                       close=monthly_averages['Close'].values.tolist(),
                      increasing=dict(line=dict(color= 'red')),
                decreasing=dict(line=dict(color= 'lightgreen')))

layout = {
    'title': 'Twitter Stocks <br> <i> After Covid </i>',
    'xaxis': {'title': 'Date',
             'rangeslider': {'visible': False}},
    'yaxis': {'title': 'Stock Price (USD$)'},
    'shapes': [{
        'x0': 0, 'x1': 1,
        'y0': max_value, 'y1': max_value, 'xref': 'paper',
        'line': {'color': 'rgb(30,30,30)', 'width': 1}
        }],
    'annotations': [{
        'x': '2020-03-15', 'y': 0.95, 'xref': 'x', 'yref': 'paper',
        'showarrow': False,
        'text': 'Peak Value = %f' %max_value
    }]
}


data = [trace]

fig = go.Figure(data=data, layout=layout)
fig.update(layout_xaxis_rangeslider_visible=True)
fig.show()


# Add 10, 50, and 200 Day moving average to Chart
df['10D_avg'] = df.Close.rolling(window=10).mean()
df['50D_avg'] = df.Close.rolling(window=50).mean()
df['200D_avg'] = df.Close.rolling(window=200).mean()

# Plot
fig = make_subplots()

colors = ['yellow', '#E6a1cf', '#6E6E6E']
avgs = ['10D_avg', '50D_avg', '200D_avg']

fig.append_trace({'x': df["Date"], 'y': df['Close'], 'type': 'scatter', 'name': 'Close', 'line': {'color': 'green'}}, 1, 1)

for col, c in zip(avgs, colors):
    fig.append_trace({'x': df["Date"], 'y': df[col], 'type': 'scatter', 'name': col, 'line': {'color': c}}, 1, 1)

fig['layout'].update(height=800, width=1000, title='Relationship between MAs and Closing Price')

fig.show()
