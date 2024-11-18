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
        self.node_mapping = {'PETR4': 0, 'VALE3': 1, 'ITUB4': 2}

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


def test_initialization(node2vec):
    assert node2vec.model.__class__.__name__ == 'Node2Vec'

def test_train(node2vec):
    loss_percent = node2vec.train()
    assert loss_percent > 0

def test_get_embeddings(node2vec):
    embeddings, node_mapping = node2vec.get_embeddings()
    assert embeddings.shape == (3, 128)
    assert node_mapping == {'PETR4': 0, 'VALE3': 1, 'ITUB4': 2}