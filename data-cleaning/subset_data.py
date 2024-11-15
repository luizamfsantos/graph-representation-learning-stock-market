import numpy as np 
import pandas as pd
import re 
from pathlib import Path

# Not shown here:
# Download the data, 
# unzip it and save it 
# in the data folder

# Read the data
folder_path = Path(__file__).parent.parent / 'data' 
file_list = ['bovespa_indexes.csv', 'bovespa_stocks.csv']
stocks_df = pd.read_csv(folder_path / file_list[1])

# Clean the data
stocks_df['Date'] = stocks_df['Date'].astype(str)\
.apply(lambda x: re.search(r'\d{4}-\d{2}-\d{2}', x)\
.group(0))
stocks_df['Date'] = pd.to_datetime(stocks_df['Date'], 
format='%Y-%m-%d')
stocks_df.dropna(subset=['Date','Symbol', 'Adj Close'], 
inplace=True)
stocks_df = stocks_df[stocks_df['Adj Close'] > 0]
stocks_df = stocks_df[stocks_df['Volume'] > 0]
stocks_df = stocks_df[['Date', 'Symbol', 'Adj Close']]
stocks_df.rename(
    columns={'Adj Close': 'adj_close',
    'Symbol': 'ticker',
    'Date': 'date'}, inplace=True
)
stocks_df = stocks_df[stocks_df.date.dt.year >= 2023]
stocks_df = stocks_df.groupby(['date', 'ticker'])\
    .mean().reset_index()
stocks_df = stocks_df.pivot(
    index='date', columns='ticker', values='adj_close'
)
stocks_df = stocks_df.dropna(
    thresh=int(stocks_df.shape[0]*0.8),axis=1)
stocks_df = stocks_df.ffill().bfill()
stocks_df = stocks_df.pct_change()
stocks_df = stocks_df.dropna()
stocks_df = (stocks_df - stocks_df.mean()) / stocks_df.std()

# Save the data
stocks_df.to_csv(folder_path / 'stocks_cleaned.csv')
stocks_df.columns.to_series().to_csv(
    folder_path / 'stocks_tickers.csv', index=False
)