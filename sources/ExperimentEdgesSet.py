from sources.ExperimentEdge import ExperimentEdge
from sources.EdgesSet import EdgesSet

class ExperimentEdgesSet(EdgesSet):
    def __init__(self):
        super().__init__()

    def weight(self):
        if not self.edges:
            return 0  # An empty set of edges has weight 0

        current_weight = 1
        for edge in self.edges:
            current_weight *= edge.get_weight()
        return current_weight

    def get_vertex_colouring(self, number_of_nodes):
        vertex_colouring = [None] * number_of_nodes
        for edge in self.edges:
            u, v = edge.get_u(), edge.get_v()
            vertex_colouring[u] = edge.get_colour(u)
            vertex_colouring[v] = edge.get_colour(v)
        return tuple(vertex_colouring)

    def to_dict(self, number_of_nodes):
        return {
            "weight": {
                "real": self.weight().real,
                "imaginary": self.weight().imag
            },
            "induced_vertex_colouring": self.get_vertex_colouring(number_of_nodes),
            "edges": [edge.to_dict() for edge in self.edges]
        }

    def add_edge(self, edge):
        if type(edge) != ExperimentEdge:
            raise TypeError("The edge must be of type ExperimentEdge.")
        self.edges.add(edge)

    def __add__(self, other):
        if type(other) == ExperimentEdge:
            new_set = ExperimentEdgesSet()
            new_set.edges = self.edges.union({other})
            return new_set
        if type(other) == ExperimentEdgesSet:
            new_set = ExperimentEdgesSet()
            new_set.edges = self.edges.union(other.edges)
            return new_set
        raise TypeError("Unsupported operand type(s) for +: 'ExperimentEdgesSet' and " + str(type(other)))

    def __str__(self):
        return str(self.edges)

    def __repr__(self):
        return str(self.edges)