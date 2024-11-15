"""
This script performs data cleaning and preprocessing on stock market data.

Steps:
1. Reads the stock market data from CSV files.
2. Cleans the data by:
    - Converting the 'Date' column to datetime format.
    - Dropping rows with missing values in 'Date', 'Symbol', and 'Adj Close' columns.
    - Filtering out rows where 'Adj Close' and 'Volume' are less than or equal to 0.
    - Renaming columns for consistency.
    - Filtering data to include only records from the year 2023 onwards.
    - Removing duplicates in a date for the same ticker by grouping data by 'date' and 'ticker' and calculating the mean.
    - Pivoting the DataFrame to have 'date' as the index and 'ticker' as columns.
    - Dropping columns with more than 20% missing values.
    - Forward-filling and backward-filling missing values.
    - Calculating the percentage change of the data.
    - Standardizing the data by subtracting the mean and dividing by the standard deviation.
3. Saves the cleaned and preprocessed data to CSV files.

Outputs:
- 'stocks_cleaned.csv': Cleaned and preprocessed stock market data.
- 'stocks_tickers.csv': List of stock tickers used in the cleaned data.
"""
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
market_index_df = pd.read_csv(folder_path / file_list[0])

# Clean the stock data
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

stocks_normalized_df = (stocks_df - stocks_df.mean()) / stocks_df.std()
stocks_normalized_df.to_csv(folder_path / 'stocks_normalized.csv')
stocks_df.columns.to_series().to_csv(
    folder_path / 'stocks_tickers.csv', index=False
)

# Clean the market index data
market_index_df['Date'] = market_index_df['Date'].astype(str)\
.apply(lambda x: re.search(r'\d{4}-\d{2}-\d{2}', x)\
.group(0))
market_index_df['Date'] = pd.to_datetime(market_index_df['Date'],
    format='%Y-%m-%d')
market_index_df = market_index_df[
    market_index_df['Symbol'] == '^BVSP'
]
market_index_df = market_index_df[['Date', 'Adj Close']]
market_index_df = market_index_df.rename(
    columns={'Adj Close': 'index',
    'Date': 'date'}
)
market_index_df.set_index('date', inplace=True)
market_index_df.dropna(inplace=True)
market_index_df = market_index_df.pct_change()

market_index_normalized_df = (
    market_index_df - market_index_df.mean()
    ) / market_index_df.std()
market_index_normalized_df.to_csv(
    folder_path / 'market_index_normalized.csv'
)

# find the intersection of the two dataframes indexes
intersection_dates = stocks_df.index.intersection(
    market_index_df.index
)
stocks_df = stocks_df.loc[intersection_dates]
market_index_df = market_index_df.loc[intersection_dates]

# Save the cleaned data
stocks_df.to_csv(folder_path / 'stocks_cleaned.csv')
market_index_df.to_csv(folder_path / 'market_index_cleaned.csv')