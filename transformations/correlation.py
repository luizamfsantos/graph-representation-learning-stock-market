import numpy as np 
import pandas as pd 
from pathlib import Path

# load data 
folder_path = Path(__file__).parent.parent / 'data'
file_path = folder_path / 'residuals.csv'
data = pd.read_csv(file_path, 
index_col='date', parse_dates=True)

# calculate correlation matrix
correlation_matrix = data.corr()

# save correlation matrix to CSV
correlation_matrix.to_csv(folder_path / 'correlation_matrix.csv')