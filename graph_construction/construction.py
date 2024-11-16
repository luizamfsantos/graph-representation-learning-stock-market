from graph_construction.visualization import visualize_graph
from graph_construction.kruskal import Graph
import pandas as pd 
from pathlib import Path


# load data
folder_path = Path(__file__).parent.parent / 'data' 
nodes = pd.read_csv(folder_path / 'stocks_tickers.csv')['ticker'].tolist()
edge_weights = pd.read_csv(folder_path / 'correlation_matrix.csv', index_col=0)


# create graph
g = Graph(nodes) # creates graph without any edges
for node_i in nodes:
    for node_j in nodes:
        if node_i != node_j:
            g += (node_i, node_j, edge_weights.loc[node_i, node_j])


# Create a minimum spanning tree
mst = g.kruskal()

# Visualize the graph and the minimum spanning tree
visualize_graph(g)
visualize_graph(g.mst)