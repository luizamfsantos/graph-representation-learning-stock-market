import pytest 
import numpy as np
import pandas as pd
from pathlib import Path
from evaluation.calculate_loss import load_data

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