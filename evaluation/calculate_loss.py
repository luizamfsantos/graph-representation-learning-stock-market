import numpy as np
import pandas as pd
from pathlib import Path

def load_data(folder_path, n_clusters):
    """Load the correlation matrix and cluster labels."""
    # Load correlation matrix
    corr_matrix = pd.read_csv(folder_path / 'correlation_matrix.csv', index_col=0)
    
    # Load cluster labels for the given number of clusters
    cluster_labels = pd.read_csv(folder_path / f'cluster_labels_{n_clusters}_clusters.csv', index_col=0)
    
    # Ensure dataframes are sorted consistently
    corr_matrix = corr_matrix.sort_index(axis=0).sort_index(axis=1)
    cluster_labels = cluster_labels.sort_index(axis=0)
    
    return corr_matrix, cluster_labels

def calculate_cluster_matrices(cluster_labels):
    """Create same-cluster and different-cluster matrices."""
    same_cluster_matrix = (
        cluster_labels.values[:, np.newaxis] == cluster_labels.values[np.newaxis, :]).astype(float).squeeze()
    diff_cluster_matrix = -1 * (same_cluster_matrix - 1).astype(int)
    return same_cluster_matrix, diff_cluster_matrix

def calculate_high_correlation_matrix(corr_matrix, rho_min):
    """Filter the correlation matrix based on the threshold rho_min."""
    high_corr_matrix = corr_matrix[corr_matrix > rho_min].values
    high_corr_matrix = np.nan_to_num(high_corr_matrix)  # Replace NaNs with 0
    return high_corr_matrix

def evaluate_loss(diff_cluster_matrix, high_corr_matrix, n_stocks):
    """Calculate the number of misses and the loss percentage."""
    assert diff_cluster_matrix.shape == high_corr_matrix.shape, \
        'Mismatch shapes between cluster matrix and correlation matrix'
    
    miss_matrix = np.logical_and(diff_cluster_matrix, high_corr_matrix).astype(int)
    misses_count = np.sum(miss_matrix)
    loss_percentage = misses_count / (n_stocks ** 2)  # misses over total edges
    
    return misses_count, loss_percentage

def main(folder_path, n_clusters_list, rho_min_list):
    """Main function to evaluate for different n_clusters and rho_min."""
    results = []
    
    for n_clusters in n_clusters_list:
        corr_matrix, cluster_labels = load_data(folder_path, n_clusters)
        n_stocks = len(cluster_labels)
        
        same_cluster_matrix, diff_cluster_matrix = calculate_cluster_matrices(cluster_labels)
        
        for rho_min in rho_min_list:
            high_corr_matrix = calculate_high_correlation_matrix(corr_matrix, rho_min)
            misses_count, loss_percentage = evaluate_loss(diff_cluster_matrix, high_corr_matrix, n_stocks)
            
            results.append({
                'n_clusters': n_clusters,
                'rho_min': rho_min,
                'misses_count': misses_count,
                'loss_percentage': loss_percentage
            })
    
    results_df = pd.DataFrame(results)
    return results_df

if __name__ == '__main__':
    # Parameters
    folder_path = Path(__file__).parent.parent / 'data'
    #n_clusters_list = [2, 3, 4, 5]  # Define the list of n_clusters to evaluate
    n_clusters_list = [3, 4, 5, 6, 7, 8, 9]  # Define the list of n_clusters to evaluate
    rho_min_list = [0.8, 0.85, 0.9, 0.95]  # Define the list of rho_min values to evaluate

    # Run evaluation
    results_df = main(folder_path, n_clusters_list, rho_min_list)

    # Print or save results
    print(results_df)
