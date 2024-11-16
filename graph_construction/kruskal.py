import numpy as np 
from graph_construction.visualization import visualize_graph

class DisjointSet:
    def __init__(self, nodes: list[str]):
        self.parent = {} # parent of the node
        self.rank = {} # rank of the node
        for node in nodes:
            self.make_set(node)

    def make_set(self, node: str):
        self.parent[node] = node # parent of itself
        self.rank[node] = 0 # rank of itself

    def find(self, node: str):
        """ Function to find the representative of the set """
        # check if node is the parent of itself
        # aka the representative of the set
        if self.parent[node] == node:
            return node
        # else call find recursively
        self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1: str, node2: str):
        """ Function to union two sets """
        # find the representative of the sets
        parent1 = self.find(node1)
        parent2 = self.find(node2)
        # if the representatives are the same
        # then the nodes are already in the same set
        if parent1 == parent2:
            return
        # else merge the sets
        if self.rank[parent1] > self.rank[parent2]:
            # merge the smaller set into the larger set
            self.parent[parent2] = parent1
        elif self.rank[parent1] < self.rank[parent2]:
            # merge the smaller set into the larger set
            self.parent[parent1] = parent2
        else:
            # equal rank, merge any set into any set
            self.parent[parent1] = parent2
            self.rank[parent2] += 1 # increase the rank of the set



class Graph:
    def __init__(
        self, 
        nodes: list[str],
        ):
        self.nodes = nodes
        self.edge_list = []

    # overload add operator to add edges
    def __add__(self, edge: tuple[str, str, float | int]):
        # check if the nodes exist
        self.check_node_exists(edge[0])
        self.check_node_exists(edge[1])
        # don't add the edge if it already exists
        if self.check_edge_exists(edge):
            return self
        self.edge_list.append(edge)
        return self

    def check_node_exists(
        self, 
        node: str):
        if node in self.nodes:
            return True
        raise ValueError(f"Edge {edge} does not exist in the graph")

    def check_edge_exists(
        self, 
        edge: tuple[str, str, int]):
        if edge in self.edge_list:
            return True
        return False

    def sort_edges(self):
        self.edge_list.sort(key=lambda x: x[2])

    def kruskal(self):
        # create a disjoint set
        disjoint_set = DisjointSet(self.nodes)
        # sort the edges
        self.sort_edges()
        # create a minimum spanning tree
        mst = []
        for edge in self.edge_list:
            node1, node2, weight = edge
            # check if the nodes are in the same set
            if disjoint_set.find(node1) != disjoint_set.find(node2):
                # add the edge to the minimum spanning tree
                mst.append(edge)
                # union the sets
                disjoint_set.union(node1, node2)
        # create a graph from the minimum spanning tree
        # to visualize the minimum spanning tree
        self.mst = Graph(self.nodes)
        for edge in mst:
            self.mst += edge
        return mst

def create_fully_connected_graph(
    nodes: list[str],
    weights: list[int],
    ):
    graph = Graph(nodes)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            graph += (nodes[i], nodes[j], weights[i][j])
    return graph


if __name__ == '__main__':
    max_weight = 10
    min_weight = 1
    nodes = ['A', 'B', 'C', 'D', 'E']
    weights = np.random.randint(
        min_weight, 
        max_weight,
        (len(nodes), len(nodes)))
    graph = create_fully_connected_graph(
        nodes=nodes,
        weights=weights,
    )
    mst = graph.kruskal()
    print(mst)
    visualize_graph(graph)
    visualize_graph(graph.mst)
    