import dash
from dash import dcc
from dash import html
import plotly.express as px
from dash import Dash, html, Input, Output, callback_context
import dash_bootstrap_components as dbc

from merlion.models.factory import ModelFactory
from merlion.utils import TimeSeries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from merlion.models.defaults import DefaultForecasterConfig, DefaultForecaster
from merlion.evaluate.forecast import ForecastMetric
from merlion.models.ensemble.combine import Mean, ModelSelector
from merlion.models.ensemble.forecast import ForecasterEnsemble, ForecasterEnsembleConfig

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pickle

def load_data(path):
    df = pd.read_csv(path, sep=';')
    df = df.rename(columns={
        'User DL Average Throughput_LTE(kB/s)' : 'dl_avg',
        'User UL Average Throughput_LTE(kB/s)' : 'ul_avg',
        'Cell DL Peak Throughput_LTE(MB/s)' : 'dl_peak',
        'Cell UL Peak Throughput_LTE(MB/s)' : 'ul_peak'
    })
    df = df[df.columns[:-1]]
    df = df.dropna()
    df['eNodeB Name'] = df['eNodeB Name'].apply(lambda x: str(x).lower())
    df['Time']=pd.to_datetime(df['Time'])
    
    filtered = df[df['eNodeB Name'].map(df['eNodeB Name'].value_counts()) == 720]
    filtered = filtered.rename(columns={'Time':'ds'})
    bts_name = filtered['eNodeB Name'].unique()
    
    return filtered, bts_name

# get data
def get_data(bts_data, bts_name, column):
    bts_data = bts_data[bts_data['eNodeB Name']==bts_name]
    bts_data_col = bts_data.rename(columns={f'{column}':'y'})
    bts_data_col = bts_data_col[['ds', 'y']]
    train_length = len(bts_data_col) - (7*24) 
    train = bts_data_col.iloc[:train_length]
    test = bts_data_col.iloc[train_length:]
    return train, test

#fit and save model
def fit_save(bts_data, bts_name, column):
    m = Prophet()
    m.fit(bts_data)
    with open(f'models/{bts_name}_{column}.pckl', 'wb') as fout:
        pickle.dump(m, fout)
        
def load_model(bts_name, column):
    with open(f'models/{bts_name}_{column}.pckl', 'rb') as fin:
        model = pickle.load(fin)
    return model

def predict(model, day):
    future = model.make_future_dataframe(periods=day)
    forecast = model.predict(future)
    return forecast

path = 'data/data throughput only.csv'
data, bts_names = load_data(path)
cols = ['dl_avg', 'ul_avg', 'dl_peak', 'ul_peak']

app = dash.Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div([
    html.H1(
        children='Predicting User Throughput Using Machine Learning',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    
    html.Div([
        html.Div([
#             dcc.Dropdown(
#                 id='model-selector',
#                 options=[{'label': i, 'value': i} for i in model_types],
#                 value=model_types[1]
#             ),
#         ], style={'width': '33%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='category-selector',
                options=[{'label': i, 'value': i} for i in cols],
                value=cols[0]
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Dropdown(
                id='bts-selector',
                options=[{'label': i, 'value': i} for i in bts_names],
                value=bts_names[0]
            ),
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    dcc.Graph(id='forecast'),
    dcc.Slider(
        id='days-slider',
        min=1,
        max=7,
        value=7,
        marks={str(i): str(i) for i in range(1, 8)},
        step=1
    ),
    html.Div([
        html.Div([
            html.H3('MAE'),
            html.H4(id='MAE')
        ],style={'width': '33%', 'display': 'inline-block'}),
        html.Div([
            html.H3('MSE'),
            html.H4(id='MSE')
        ],style={'width': '66%', 'float': 'right', 'display': 'inline-block'})
    
     ])
])
])
@app.callback(
    Output('forecast', 'figure'),
    Output('MAE', 'children'),
    Output('MSE', 'children'),
    #Input('model-selector', 'value'),
    Input('category-selector', 'value'),
    Input('bts-selector', 'value'),
    Input('days-slider', 'value')
)

def update_graph(data_type, bts_name, days):
    model = load_model(bts_name, data_type)
    forecast = predict(model, days*24)
    forecast = forecast.tail(days*24)
    _, test_data = get_data(data, bts_name, data_type)
    test_data = test_data.iloc[:days*24]
    test_data['yhat'] = forecast['yhat'].values
    fig = px.line(test_data, x='ds', y=['y', 'yhat'], labels={
                         "y": "Actual",
                         "yhat": "Predicted",
                     })
    mae = mean_absolute_error(test_data['y'], test_data['yhat'])
    mse = mean_squared_error(test_data['y'], test_data['yhat'])
    return fig, round(mae,3), round(mse,3)


if __name__ == '__main__':
    app.run_server(debug=False)