"""
This script performs linear regression on stock prices against a market index and calculates the residuals.
Residuals is the non-systematic part of the stock return that is not explained by the market index.

Modules:
    pandas
    numpy
    pathlib
    sklearn.linear_model

Functions:
    None

Variables:
    folder_path (Path): The path to the data folder.
    file_list (list): List of filenames for stock and market index data.
    stocks_df (DataFrame): DataFrame containing stock prices.
    market_index_df (DataFrame): DataFrame containing market index values.
    merged_df (DataFrame): DataFrame containing merged stock and market index data.
    residuals_df (DataFrame): DataFrame to store residuals of the regression.

Procedure:
    1. Load stock and market index data from CSV files.
    2. Merge the stock and market index data on the date index.
    3. Initialize an empty DataFrame to store residuals.
    4. For each stock, perform linear regression against the market index.
    5. Calculate residuals by subtracting the predicted stock return from the actual stock return.
    6. Save the residuals to a CSV file.
"""
import pandas as pd 
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

# load data
folder_path = Path(__file__).parent.parent / 'data'
file_list = ['stocks_cleaned.csv', 'market_index_cleaned.csv']
stocks_df = pd.read_csv(folder_path / file_list[0],
index_col='date', parse_dates=True)
market_index_df = pd.read_csv(folder_path / file_list[1],
index_col='date', parse_dates=True)
merged_df = pd.merge(stocks_df, 
market_index_df, left_index=True, right_index=True)

residuals_df = pd.DataFrame(
    index=stocks_df.index, columns=stocks_df.columns)

for stock in stocks_df.columns:
    df = merged_df[[stock, 'market_index']].replace(0, np.nan).dropna()
    model = LinearRegression()
    model.fit(df['market_index'].values.reshape(-1, 1), df[stock])
    residuals_df[stock] = stocks_df[stock] - model.predict(
        market_index_df.values.reshape(-1, 1))

residuals_df.to_csv(folder_path / 'residuals.csv')