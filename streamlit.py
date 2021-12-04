import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pickle

import seaborn as sns

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
        
def load_model(bts_name, column):
    with open(f'models/{bts_name}_{column}.pckl', 'rb') as fin:
        model = pickle.load(fin)
    return model

def predict(model, day):
    future = model.make_future_dataframe(periods=day)
    forecast = model.predict(future)
    return forecast

def filter_data(bts_name, data_type, days):
    model = load_model(bts_name, data_type)
    forecast = predict(model, days*24)
    forecast = forecast.tail(days*24)
    _, test_data = get_data(data, bts_name, data_type)
    test_data = test_data.iloc[:days*24]
    test_data = test_data.rename(columns={'y':'value'})
    test_data['label'] = 'actual'
    
    forecast_df = pd.DataFrame(data={'value':forecast['yhat'].values, 'ds':test_data['ds'].values})
    forecast_df['label'] = 'predicted' 
    
    return test_data.append(forecast_df)
    
def make_fig(test_data):
    fig ,ax = plt.subplots() 
    sns.lineplot(data=test_data, x='ds', y='value', hue='label', ax=ax)
#     fig = px.line(test_data, x='ds', y=['y', 'yhat'], labels={
#                          "y": "Actual",
#                          "yhat": "Predicted",
#                      })
    return fig

path = 'data/data throughput only.csv'
data, bts_names = load_data(path)
cols = ['dl_avg', 'ul_avg', 'dl_peak', 'ul_peak']


##################################Layout##################################
##################################row 1##################################
st.title('User Throughput Prediction')
#row 1
lc_1, rc_1 = st.columns(2)
with lc_1:
    select_data = st.selectbox(
        'Which user throughput you want to predict?',
         cols)
with rc_1:
    select_bts = st.selectbox(
        'Which bts you want to see?',
        bts_names
    )
##################################row 2##################################
select_day = st.slider(
    'days to predict',
    1,
    5,
    7
)
##################################Select Data BTS with specific data type and the model prediction##################################
selected_data = filter_data(select_bts, select_data, select_day)
##################################row 3##################################
with st.container():
    st.title(f'Line chart {select_data} for {select_bts}')
    fig = make_fig(selected_data)
    st.pyplot(fig)
##################################row 4##################################
lc_4, rc_4 = st.columns(2)
with lc_4:
    with st.container():
        st.header('MAE')
        mae = mean_absolute_error(selected_data[selected_data['label']=='actual']['value'], selected_data[selected_data['label']=='predicted']['value'])
        st.subheader(f'{mae}')
        
        
with rc_4:
    with st.container():
        st.header('MSE')
        mse = mean_squared_error(selected_data[selected_data['label']=='actual']['value'], selected_data[selected_data['label']=='predicted']['value'])
        st.subheader(f'{mse}')
