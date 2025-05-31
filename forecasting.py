# forecasting.py
import pandas as pd
from prophet import Prophet

def preprocess_forecast_data(df):
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    monthly_sales = df.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum().reset_index()
    monthly_sales.columns = ['ds', 'y']  # Prophet needs these column names
    return monthly_sales

def forecast_sales(df, periods=6):
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq='M')
    forecast = m.predict(future)
    return forecast
