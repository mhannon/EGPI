from sources.Edge import Edge


class MultiGraph:
    def __init__(self):
        self.adjacency = {}
        self.number_of_nodes = 0
        self.number_of_edges = 0

    def add_node(self, node: int):
        if node not in self.adjacency:
            self.adjacency[node] = {}
            self.number_of_nodes += 1

    def add_nodes_from(self, nodes):
        for node in nodes:
            self.add_node(node)

    def add_edge(self, u: int, v: int):
        assert self.has_node(u) and self.has_node(v), \
            "Nodes must be in the graph before adding an edge."

        if v not in self.adjacency[u]:
            self.adjacency[u][v] = []
        if u not in self.adjacency[v]:
            self.adjacency[v][u] = []
        edge_id = len(self.adjacency[u][v])
        edge = Edge(u, v, edge_id)
        self.adjacency[u][v].append(edge)
        self.adjacency[v][u].append(edge)
        self.number_of_edges += 1

    def add_edges_from(self, edges):
        for edge in edges:
            self.add_edge(edge.get_u(), edge.get_v())

    def remove_edge(self, edge: Edge):
        u, v = edge.get_u(), edge.get_v()
        if self.has_edge(edge):
            self.adjacency[u][v].remove(edge)
            self.adjacency[v][u].remove(edge)

            if not self.adjacency[u][v]:
                del self.adjacency[u][v]
            if not self.adjacency[v][u]:
                del self.adjacency[v][u]
            self.number_of_edges -= 1

    def remove_node(self, node: int):
        removed_edges = []
        if self.has_node(node):
            for neighbour in self.neighbours(node):
                removed_edges += self.edges(node, neighbour)
                del self.adjacency[neighbour][node]
            del self.adjacency[node]  # TODO: Check if this is correct
        self.number_of_nodes -= 1
        self.number_of_edges -= len(removed_edges)
        return removed_edges

    def neighbours(self, node: int):
        if node in self.adjacency:
            return list(self.adjacency[node].keys())
        return []

    def edges(self, u: int, v: int):
        if u in self.adjacency and v in self.adjacency[u]:
            return self.adjacency[u][v]
        return []

    def has_node(self, node: int):
        return node in self.adjacency

    def has_edge(self, edge: Edge):
        u, v = edge.get_u(), edge.get_v()
        if u in self.adjacency and v in self.adjacency[u]:
            return edge in self.adjacency[u][v]
        return False

    def has_edges(self, u: int, v: int):
        return u in self.adjacency and v in self.adjacency[u]

    def __copy__(self):
        new_graph = MultiGraph()
        new_graph.adjacency = self.adjacency.copy()
        new_graph.number_of_nodes = self.number_of_nodes
        new_graph.number_of_edges = self.number_of_edges
        return new_graph

    def __eq__(self, other):
        return self.adjacency == other.adjacency

    def __str__(self):
        return str(self.adjacency)

    def __repr__(self):
        return str(self)
