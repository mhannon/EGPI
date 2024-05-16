import itertools
import math
import os

from sources.ExperimentEdge import ExperimentEdge
from sources.ExperimentEdgesSet import ExperimentEdgesSet
from sources.MultiGraph import MultiGraph


def get_coordinates_of_intermediate_point(p1, p2, edge_id, number_of_edges):
    """
    Given two points, returns the point that is in the middle of the segment that joins them.
    :param p1: tuple representing the coordinates of the first point.
    :param p2: tuple representing the coordinates of the second point.
    :param edge_id: ID of the edge.
    :param number_of_edges: number of edges between the two points.
    :return: tuple representing the coordinates of the intermediate point.
    """
    normalized_perpendicular_vector = ((p2[1] - p1[1]) / math.sqrt((p2[1] - p1[1]) ** 2 + (p1[0] - p2[0]) ** 2),
                                       (p1[0] - p2[0]) / math.sqrt((p2[1] - p1[1]) ** 2 + (p1[0] - p2[0]) ** 2))
    mean_x = (p1[0] + p2[0]) / 2        # basic x-coordinate
    mean_y = (p1[1] + p2[1]) / 2        # basic y-coordinate
    space_between_edges = 1.0 / 2.0     # TODO : change this to a parameter
    increment = number_of_edges // 2 - edge_id
    mean_x += normalized_perpendicular_vector[0] * space_between_edges * increment
    mean_y += normalized_perpendicular_vector[1] * space_between_edges * increment
    return mean_x, mean_y


class ExperimentGraph(MultiGraph):
    def __init__(self):
        super().__init__()

    def add_edge(self, u: int, v: int, u_colour, v_colour, weight: complex):
        assert self.has_node(u) and self.has_node(v), \
            "Nodes must be in the graph before adding an edge."
        assert u != v, \
            "Self-loops are not allowed."

        if v not in self.adjacency[u]:
            self.adjacency[u][v] = []
        if u not in self.adjacency[v]:
            self.adjacency[v][u] = []

        edge_id = len(self.adjacency[u][v])
        edge = ExperimentEdge(u, v, edge_id, weight, u_colour, v_colour)
        self.number_of_edges += 1
        self.adjacency[u][v].append(edge)
        self.adjacency[v][u].append(edge)

    def add_edges_from(self, edges):
        for edge in edges:
            self.add_edge(edge.get_u(), edge.get_v(), edge.get_u_colour(), edge.get_v_colour(), edge.get_weight())

    def get_perfect_matchings(self, remaining_nodes=None) -> list[ExperimentEdgesSet]:
        # First time we call the function
        if remaining_nodes is None:
            remaining_nodes = set(range(self.number_of_nodes))

        # Base case: no nodes left
        if not remaining_nodes:
            return [ExperimentEdgesSet()]

        # Recursive case
        perfect_matchings = []
        u = remaining_nodes.pop()
        for v in remaining_nodes:
            if self.has_edges(u, v):
                sub_perfect_matchings = self.get_perfect_matchings(remaining_nodes - {v})
                for matching in sub_perfect_matchings:
                    for edge in self.edges(u, v):
                        new_matching = matching + edge
                        perfect_matchings.append(new_matching)
        return perfect_matchings

    def get_feasible_vertex_colourings_weights(self):
        """
        Returns a dictionary of all feasible vertex colourings and their weights.
        :return: dict
        """
        feasible_vertex_colourings = dict()
        for perfect_matching in self.get_perfect_matchings():
            vertex_colouring = perfect_matching.get_vertex_colouring(self.number_of_nodes)
            if vertex_colouring in feasible_vertex_colourings:
                feasible_vertex_colourings[vertex_colouring] += perfect_matching.weight()
            else:
                feasible_vertex_colourings[vertex_colouring] = perfect_matching.weight()
        return feasible_vertex_colourings

    def is_perfectly_monochromatic(self):
        """
        Returns True if the graph is perfectly monochromatic, False otherwise.
        :return: bool
        """
        feasible_vertex_colourings_weights = self.get_feasible_vertex_colourings_weights()
        for vertex_colouring in feasible_vertex_colourings_weights:
            if len(set(vertex_colouring)) == 1:  # The vertex colouring is monochromatic
                if feasible_vertex_colourings_weights[vertex_colouring] != 1:
                    return False
            else:
                if feasible_vertex_colourings_weights[vertex_colouring] != 0:
                    return False
        return True

    def get_weighted_matching_index(self):
        """
        Returns the weighted matching index of the graph.
        :return: int
        """
        if not self.is_perfectly_monochromatic():
            return 0  # If the graph is not perfectly monochromatic, the index is 0

        c = 0
        for vertex_colouring in self.get_feasible_vertex_colourings_weights():
            if len(set(vertex_colouring)) == 1:  # The vertex colouring is monochromatic
                c += 1
        return c

    def generate_latex(self):
        """
        Generates a LaTeX representation of the graph.
        :return: str
        """
        latex = r"""
        \documentclass{standalone}
        \usepackage{tikz}
        \begin{document}
        \begin{tikzpicture}
        \tikzstyle{every node}=[font=\tiny]
        """
        node_positions = self.get_nodes_coordinates(4.0)

        for u, v in itertools.combinations(range(self.number_of_nodes), 2):
            edges = self.edges(u, v)
            number_of_edges = len(edges)
            for edge in edges:
                x_u, y_u = node_positions[u]
                x_m, y_m = get_coordinates_of_intermediate_point(node_positions[u], node_positions[v], edge.get_id(), number_of_edges)
                x_v, y_v = node_positions[v]

                u_colour = edge.get_u_colour()
                v_colour = edge.get_v_colour()
                c_weight = edge.weight_to_string()

                latex += "\\draw [style=thin, color=" + u_colour + "] (" + str(x_u) + "," + str(y_u) + ") to (" + str(x_m) + "," + str(y_m) + ");\n"
                latex += "\\draw [style=thin, color=" + v_colour + "] (" + str(x_m) + "," + str(y_m) + ") to (" + str(x_v) + "," + str(y_v) + ");\n"
                latex += "\\node [style=circle, draw=none] at (" + str(x_m) + "," + str(y_m) + ") {" + c_weight + "};\n"

        for node in range(self.number_of_nodes):
            x, y = node_positions[node]
            latex += "\\node [style=circle, fill=white, draw=black] (" + str(node) + ") at (" + str(x) + "," + str(y) + ") {" + str(node) + "};\n"

        latex += "\\end{tikzpicture}\n"
        latex += "\\end{document}"
        return latex

    def to_pdf(self, folder: str, filename: str):
        """
        Generates a PDF file with the LaTeX representation of the graph.
        :param folder: folder where the PDF file will be stored.
        :param filename: name of the file to generate (without the .tex).
        :return: None
        """
        path = folder + "/" + filename
        print(os.path.abspath(os.curdir))
        with open(f"{path}.tex", "w") as f:
            f.write(self.generate_latex())
        os.system(f"pdflatex {path}.tex" + " -output-directory=" + folder)
        os.remove(f"{path}.log")
        os.remove(f"{path}.aux")

    def get_nodes_coordinates(self, space_between_nodes):
        """
        Returns the coordinates of the nodes in the graph.
        :param space_between_nodes: distance between nodes.
        :return: tuple of tuples representing the coordinates of the nodes.
        """
        assert space_between_nodes > 0, "The space between nodes must be positive."

        # Calculate the angle between each point
        angle_increment = 2 * math.pi / self.number_of_nodes
        radius = (space_between_nodes / 2) / math.tan(angle_increment / 2)

        # Initialize the list to store coordinates
        coordinates = []

        # Generate coordinates for each point
        for i in range(self.number_of_nodes):
            x = radius * math.cos(i * angle_increment)
            y = radius * math.sin(i * angle_increment)
            coordinates.append((x, y))

        return tuple(coordinates)

    def __str__(self):
        return str(self.adjacency)
