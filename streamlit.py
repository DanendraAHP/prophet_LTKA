import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
    bts_data_col = bts_data_col.set_index('ds')
    bts_data_col.index = pd.DatetimeIndex(bts_data_col.index).to_period('H')
    return bts_data_col

def forecast(bts_data, start, end):
    model=sm.tsa.statespace.SARIMAX(bts_data['y'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
    results=model.fit()
    bts_data['forecast']=results.predict(start=start*24,end=end*24,dynamic=True)
    return bts_data

def make_fig(bts_data):
    fig ,ax = plt.subplots() 
    bts_data[['y','forecast']].plot(figsize=(12,8), ax=ax)
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
start_date, end_date = st.select_slider(
    'Select a range of day to predict',
    options=[i for i in range(1, 31)],
    value=(25, 30)
)
##################################Select Data BTS with specific data type and the model prediction##################################
selected_data = get_data(data, select_bts, select_data)
selected_data = forecast(selected_data, start_date, end_date)
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
