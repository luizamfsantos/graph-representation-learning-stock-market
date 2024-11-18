import numpy as np 
import pandas as pd 
from pathlib import Path

# load residual correlation matrix
folder_path = Path(__file__).parent.parent / 'data'
corr_matrix = pd.read_csv(
    folder_path / 'correlation_matrix.csv',
    index_col=0)
corr_matrix = corr_matrix.sort_index(axis=0).sort_index(axis=1)

# load labels
n_clusters = 3 # TODO: loop over the files for different cluster numbers
cluster_labels = pd.read_csv(
    folder_path / f'cluster_labels_{n_clusters}_clusters.csv',
    index_col=0)
cluster_labels = cluster_labels.sort_index(axis=0)

# TODO: ensure that both df are sorted by index/col in the same way


# get stocks in different clusters
n_stocks = len(cluster_labels)
assert len(cluster_labels) == len(corr_matrix), \
    'Check size of cluster labels and correlation matrix'
same_cluster_matrix = (
    cluster_labels.values[:, np.newaxis] == cluster_labels.values[np.newaxis, :]).astype(float).squeeze()
diff_cluster_matrix = -1*(same_cluster_matrix-1).astype(int)

# get high correlation matrix
rho_min = 0.9 # TODO: do this for different values of rho_min
high_corr_matrix = corr_matrix[corr_matrix > rho_min].values
high_corr_matrix = np.nan_to_num(high_corr_matrix)

# get misses/loss
assert diff_cluster_matrix.shape == high_corr_matrix.shape, \
    'Mismatch shapes between cluster matrix and correlation matrix'
miss_matrix = np.logical_and(diff_cluster_matrix, high_corr_matrix)
miss_matrix = miss_matrix.astype(int)
misses_count = np.sum(miss_matrix)
loss_percentage = misses_count/(n_stocks**2) # misses over total edges