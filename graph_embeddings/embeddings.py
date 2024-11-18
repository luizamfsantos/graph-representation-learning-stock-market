from torch_geometric.nn import Node2Vec
import torch 
import numpy as np 

class Embeddings:
    def __init__(self, model, training_data, **kwargs):
        self.kwargs = kwargs
        self.training_data = training_data
        self.get_model(model)
        

    def get_model(self, model):
        if model.lower() == 'node2vec':
            self.model = Node2Vec(
                edge_index = self.training_data.edge_index,
                embedding_dim = self.kwargs.get('embedding_dim', 128),
                walk_length = self.kwargs.get('walk_length', 80),
                context_size = self.kwargs.get('context_size', 20),
                walks_per_node = self.kwargs.get('walks_per_node', 10),
                num_negative_samples = self.kwargs.get('num_negative_samples', 1),
                p = self.kwargs.get('p', 0.8),
                q = self.kwargs.get('q', 0.2),
                sparse=self.kwargs.get('sparse', True),
            )
        elif model.lower() == 'deepwalk':
            # the same as node2vec but unbiased 
            # so p = q = 1
            self.model = Node2Vec(
                edge_index = self.training_data.edge_index,
                embedding_dim = self.kwargs.get('embedding_dim', 128),
                walk_length = self.kwargs.get('walk_length', 80),
                context_size = self.kwargs.get('context_size', 20),
                walks_per_node = self.kwargs.get('walks_per_node', 10),
                num_negative_samples = self.kwargs.get('num_negative_samples', 1),
                p = 1,
                q = 1,
                sparse=self.kwargs.get('sparse', True),
            )
        else:
            raise NotImplementedError(f"Model {model} not implemented")

    def train(self):
        loader = self.model.loader(batch_size=128, shuffle=True)
        self.model.train()
        optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)
        total_loss = 0
        for positive_random_walks, negative_random_walks in loader:
            optimizer.zero_grad()
            loss = self.model.loss(positive_random_walks, negative_random_walks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss/len(loader)

    def get_embeddings(self):
        self.model.eval()
        with torch.no_grad():
            all_embeddings = self.model(
                torch.arange(
                    self.training_data.edge_index.max()+1))

        embeddings_np = all_embeddings.numpy()
        return embeddings_np, self.training_data.node_mapping
