import networkx as nx
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from pathlib import Path


def create_colored_graph(cluster_file_path, edgelist_file_path, figsize=(12,8)):
    cluster_df = pd.read_csv(cluster_file_path, index_col=0)
    
    G = nx.read_edgelist(edgelist_file_path, data=(('distances', float),))

    # Round edge distances (aka weights)
    for u, v, d in G.edges(data=True):
        d['distances'] = round(d['distances'], 2)
    
    # Create color map
    n_clusters = cluster_df['cluster_label'].nunique()
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    color_dict = dict(zip(range(n_clusters), colors))
    cluster_df['color'] = cluster_df['cluster_label'].map(color_dict)

    # Create the visualization
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, k = 1/np.sqrt(cluster_df.shape[0]), seed=42)

    # Draw nodes
    node_colors = [cluster_df.loc[node, 'color'] for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)

    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    edge_labels = nx.get_edge_attributes(G, 'distances')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    plt.title(f'Network Graph with {n_clusters} Clusters')
    plt.axis('off')

    return G

if __name__ == '__main__':
    folder_path = Path(__file__).parent.parent / 'data'
    for i in range(3,10):
        G = create_colored_graph(
            cluster_file_path=folder_path / f'cluster_labels_{i}_clusters.csv',
            edgelist_file_path= folder_path / 'mst.edgelist')
        plt.show()