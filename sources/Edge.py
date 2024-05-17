class Edge:
    def __init__(self, u: int, v: int, edge_id: int):
        self.u = u
        self.v = v
        self.id = edge_id

    def get_u(self):
        return self.u

    def get_v(self):
        return self.v

    def get_id(self):
        return self.id

    def __copy__(self):
        return Edge(self.u, self.v, self.id)

    def __eq__(self, other):
        return self.u == other.u and self.v == other.v and self.id == other.id

    def __hash__(self):
        return hash((self.u, self.v))

    def __str__(self):
        return f"({self.u}--{self.v})"

    def __repr__(self):
        return str(self)
