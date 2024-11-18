import pytest 
import numpy as np
from clustering.kmeans import cluster_embeddings


@pytest.fixture
def embeddings():
    return np.random.rand(10, 5) # num_nodes x embedding_dim

@pytest.mark.parametrize("n_clusters", [3, 5, 10])
def test_cluster_embeddings(embeddings, n_clusters):
    cluster_labels, kmeans = cluster_embeddings(embeddings, n_clusters=n_clusters)
    assert len(cluster_labels) == 10
    assert len(kmeans.cluster_centers_) == n_clusters
    assert kmeans.n_clusters == n_clusters
    assert kmeans.n_init == 10
    assert kmeans.random_state == 0
    assert kmeans.labels_.shape == cluster_labels.shape
    assert kmeans.cluster_centers_.shape[1] == embeddings.shape[1]
    assert kmeans.cluster_centers_.shape[0] == n_clusters
    assert np.all(np.isin(np.unique(cluster_labels), np.arange(n_clusters)))
    assert np.all(np.isin(np.unique(kmeans.labels_), np.arange(n_clusters)))