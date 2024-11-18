import torch
import pandas as pd 
from clustering.kmeans import cluster_embeddings
from graph_embeddings.embeddings import Embeddings
from transformations.pytorch_datasets import StockGraphDataset


def main():
    # TODO: add the ingestion/transformation/ logic from the
    # individual files to this file
    # load data
    dataset = StockGraphDataset(
        root='data',
        edgelist_file='mst.edgelist'
    )
    data = dataset[0]

    # generate embeddings
    node2vec = Embeddings(
        'node2vec',
        training_data=data,
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        walks_per_node=20
    )
    total_loss = node2vec.train()
    print(f'Total loss: {total_loss}')  # TODO: replace print for logs
    embeddings_np, node_mapping = node2vec.get_embeddings()

    # get clusters
    for n_clusters in range(3, 10):
        cluster_labels, kmeans = cluster_embeddings(
            embeddings_np,
            n_clusters=n_clusters)

        # get ticker names
        reverse_mapping = {idx: node for node, idx in node_mapping.items()}
        clustered_nodes = {
            reverse_mapping[idx]: cluster
            for idx, cluster in enumerate(cluster_labels)
        }

        # Save clusters
        df = pd.DataFrame.from_dict(clustered_nodes,
                                    orient='index', columns=['cluster_label'])
        df.to_csv(f'data/cluster_labels_{n_clusters}_clusters.csv')
    # TODO: calculate loss
    return clustered_nodes


if __name__ == '__main__':
    clustered_nodes = main()
