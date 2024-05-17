from sources.Edge import Edge


class EdgesSet:
    def __init__(self):
        self.edges = set()

    def add_edge(self, edge):
        if type(edge) != Edge:
            raise TypeError("The edge must be of type Edge.")
        self.edges.add(edge)

    def remove_edge(self, edge: Edge):
        self.edges.remove(edge)

    def __add__(self, other):
        if type(other) == Edge:
            new_set = EdgesSet()
            new_set.edges = self.edges.union({other})
            return new_set
        if type(other) == EdgesSet:
            new_set = EdgesSet()
            new_set.edges = self.edges.union(other.edges)
            return new_set
        raise TypeError("Unsupported operand type(s) for +: 'ExperimentEdgesSet' and " + str(type(other)))

    def __iter__(self):
        return iter(self.edges)

    def __len__(self):
        return len(self.edges)

    def __contains__(self, edge):
        return edge in self.edges

    def __eq__(self, other):
        return self.edges == other.edges

    def __str__(self):
        return str(self.edges)

    def __repr__(self):
        return str(self.edges)