import os
import pandas as pd 
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from typing import Optional, Callable


class StockGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,  # Path to the dataset directory
        edgelist_file: str,  # Path to the edgelist file
        transform: Optional[Callable] = None,  # Data transformation
        pre_transform: Optional[Callable] = None,  # Pre-transformation
    ):
        """ 
        Custom dataset for stock market graph data

        Args:
            root (str): Path to the dataset directory
            edgelist_file (str): Path to the edgelist file
            transform (Optional[Callable], optional): Data transformation. Defaults to None.
            pre_transform (Optional[Callable], optional): Pre-transformation. Defaults to None.
        """
        self.edgelist_file = edgelist_file
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.edgelist_file]

    @property
    def processed_file_names(self):
        basename = os.path.basename(self.edgelist_file)
        return [f"stock_graph_{basename}.pt"]

    def download(self):
        # no download required since we have a local file
        pass

    def process(self):
        # Read data from `edgelist_file`
        df = pd.read_csv(
            self.raw_paths[0],
            sep=' ',
            header=None,
            names=['source', 'target', 'weight']
        )

        # Create dictionary mapping node names to indices
        unique_nodes = np.unique(df[['source', 'target']])
        node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}

        # Convert edges to tensor format
        edge_index = torch.tensor(
            [[node_to_idx[source], node_to_idx[target]]
            for source, target in zip(df['source'], df['target'])],
            dtype=torch.long).t()

        # Convert weights to PyTorch tensor
        edge_attr = torch.tensor(df['weight'].values,
         dtype=torch.float).reshape(-1, 1)
        
        num_nodes = len(unique_nodes)
        x = torch.arange(num_nodes, 
        dtype=torch.float).reshape(-1, 1)

        # Create the data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )

        # Store node mapping for later use
        data.node_mapping = node_to_idx

        # If a pre-transform is provided, apply it
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save the data object
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.edgelist_file})"

    def __str__(self) -> str:
        return self.__repr__()

if __name__ == '__main__':
    dataset = StockGraphDataset(
        root='data',
        edgelist_file='mst.edgelist'
    )
    print(dataset)
    print(dataset[0])