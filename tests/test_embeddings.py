import torch
import pytest
from graph_embeddings.embeddings import (
    Embeddings,
)


class RandomData:
    def __init__(self):
        self.edge_index = torch.tensor([
            [0, 1, 2, 1],  # source nodes
            [1, 0, 1, 2]  # target nodes
        ])

@pytest.fixture
def data():
    return RandomData()

@pytest.fixture
def node2vec(data):
    return Embeddings(
        'node2vec',
        training_data=data,
        embedding_dim=128,
        walk_length=80,
        context_size=20,
        walks_per_node=10)


def test_node2vec(node2vec):
    assert node2vec.model.__class__.__name__ == 'Node2Vec'
