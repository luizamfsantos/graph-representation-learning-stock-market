import pytest 
import numpy as np
import pandas as pd
from pathlib import Path
from evaluation.calculate_loss import (
    load_data,
    calculate_cluster_matrices,
)

@pytest.fixture
def folder_path():
    return Path(__file__).parent.parent / 'data'


@pytest.fixture
def corr_matrix(folder_path):
    n_clusters = 3
    return load_data(folder_path, n_clusters)[0]

@pytest.fixture
def cluster_labels(folder_path):
    n_clusters = 3
    return load_data(folder_path, n_clusters)[1]


@pytest.fixture
def sample_index():
    return pd.Index(['A', 'B', 'C', 'D', 'E'])

@pytest.fixture
def sample_corr_matrix(sample_index):
    values = np.array([
        [1.0, -0.5, 0.3, 0.1, 0.2],
        [-0.5, 1.0, -0.4, 0.2, 0.1],
        [0.3, -0.4, 1.0, -0.6, 0.5],
        [0.1, 0.2, -0.6, 1.0, -0.7],
        [0.2, 0.1, 0.5, -0.7, 1.0]
    ])
    return pd.DataFrame(
        values,
        index=sample_index,
        columns=sample_index
    )

@pytest.fixture
def sample_cluster_labels(sample_index):
    return pd.DataFrame(
        np.array([0, 2, 0, 1, 2]),
        index=sample_index,
        columns=['cluster_label']
    )

def test_corr_matrix_shape(corr_matrix):
    assert corr_matrix.shape[0] > 0

def test_corr_matrix_type(corr_matrix):
    assert isinstance(corr_matrix, pd.DataFrame)

def test_cluster_labels_shape(cluster_labels):
    assert len(cluster_labels) > 0

def test_cluster_labels_type(cluster_labels):
    assert isinstance(cluster_labels, pd.DataFrame)

def test_index_match(corr_matrix, cluster_labels):
    assert corr_matrix.index.equals(cluster_labels.index)
    assert corr_matrix.columns.equals(cluster_labels.index)

def test_cluster_labels_unique(cluster_labels):
    assert cluster_labels.index.is_unique

def test_corr_matrix_unique(corr_matrix):
    assert corr_matrix.index.is_unique
    assert corr_matrix.columns.is_unique

def test_calculate_cluster_matrices(sample_cluster_labels, sample_index):
    same_cluster_matrix, diff_cluster_matrix = calculate_cluster_matrices(sample_cluster_labels)
    assert same_cluster_matrix.shape == diff_cluster_matrix.shape
    assert same_cluster_matrix.shape[0] == same_cluster_matrix.shape[1]
    for i in range(same_cluster_matrix.shape[0]):
        assert same_cluster_matrix[i][i] == 1
        assert diff_cluster_matrix[i][i] == 0
    assert same_cluster_matrix.shape[0] == len(sample_index)