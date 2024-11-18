import networkx as nx
import matplotlib.pyplot as plt

def networkx_graph(graph, round_values=False):
    G = nx.Graph()
    for edge in graph.edge_list:
        if round_values:
            G.add_edge(edge[0], edge[1], weight=round(edge[2], 2))
        else:
            G.add_edge(edge[0], edge[1], weight=edge[2])
    return G

def visualize_graph(graph, round_values=False):
    G = networkx_graph(graph, round_values)
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nx.draw(G, pos, with_labels=True)
    plt.show()