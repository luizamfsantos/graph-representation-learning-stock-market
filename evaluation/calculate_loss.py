import numpy as np 
import pandas as pd 
from pathlib import Path

# load residual correlation matrix
folder_path = Path(__file__).parent.parent / 'data'
corr_matrix = pd.read_csv(
    folder_path / 'correlation_matrix.csv',
    index_col=0)

# load labels
n_clusters = 3
cluster_labels = pd.read_csv(
    folder_path / f'cluster_labels_{n_clusters}_clusters.csv',
    index_col=0)


# get stocks in different clusters
n_stocks = len(cluster_labels)
assert len(cluster_labels) == len(corr_matrix), \
    'Check size of cluster labels and correlation matrix'
# inefficient way: 
# same_cluster_matrix = np.zeros((n_stocks, n_stocks))
# edge_list = cluster_labels.index.tolist()
# for idx, i in enumerate(edge_list):
#     cluster_i = cluster_labels.loc[i]
#     for idx, j in enumerate(edge_list):
#         cluster_j = cluster_labels.loc[j]
#         if cluster_i == cluster_j:
#             same_cluster_matrix[i,j] = 1
same_cluster_matrix = (
    cluster_labels.values[:, np.newaxis] == cluster_labels.values[np.newaxis, :]).astype(float).squeeze()
diff_cluster_matrix = -1*(same_cluster_matrix-1).astype(int)

