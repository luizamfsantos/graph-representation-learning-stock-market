import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(graph):
    G = nx.Graph()
    for edge in graph.edge_list:
        G.add_edge(edge[0], edge[1], weight=edge[2])
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nx.draw(G, pos, with_labels=True)
    plt.show()