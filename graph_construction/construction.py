from graph_construction.visualization import visualize_graph
from graph_construction.kruskal import Graph
import pandas as pd 
from pathlib import Path
import pickle


# load data
folder_path = Path(__file__).parent.parent / 'data' 
nodes = pd.read_csv(folder_path / 'stocks_tickers.csv')['ticker'].tolist()
edge_weights = pd.read_csv(folder_path / 'distance_matrix.csv', index_col=0)


# create graph
g = Graph(nodes) # creates graph without any edges
for node_i in nodes:
    for node_j in nodes:
        if node_i != node_j:
            g += (node_i, node_j, edge_weights.loc[node_i, node_j])


# Create a minimum spanning tree
mst = g.kruskal()

# Visualize the graph and the minimum spanning tree
visualize_graph(g, round_values=True)
visualize_graph(g.mst, round_values=True)

# Save pickle graph
with open(folder_path / 'graph.pkl', 'wb') as f:
    pickle.dump(g, f)

# Subset for better visualization
subset = ['ITUB4', 'LUPA3', 'M1TA34', 'ORCL34', 'HYPE3', 'GOGL35', 'FLRY3', 'EGIE3', 'DMVF3', 'COCA34', 'BHIA3', 'AERI3', 'PETR4']
assert len(subset) == len([node for node in subset if node in nodes]), 'The subset contains nodes that are not in the graph'
g2 = Graph(subset)
for node_i in subset:
    for node_j in subset:
        if node_i != node_j:
            g2 += (node_i, node_j, edge_weights.loc[node_i, node_j])

mst2 = g2.kruskal()
visualize_graph(g2, round_values=True)
visualize_graph(g2.mst, round_values=True)