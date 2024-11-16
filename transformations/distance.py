import pandas as pd 
from pathlib import Path

# load data
folder_path = Path(__file__).parent.parent / 'data' 
corr_matrix = pd.read_csv(folder_path / 'correlation_matrix.csv', index_col=0)

# transform the correlation matrix
# using the formula 2*(1-x)
distance_matrix = corr_matrix.apply(lambda x: 2*(1-x))

# save the distance matrix
distance_matrix.to_csv(folder_path / 'distance_matrix.csv')