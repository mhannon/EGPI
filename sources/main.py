"""
Author: Merlin Hannon
Date: 2024-05-07
"""


import itertools
import math
import os
from copy import deepcopy
import networkx as nx


def is_monochromatic(vertex_colouring):
    """
    Given a vertex colouring, returns True if all vertices have the same colour, False otherwise.
    :param vertex_colouring: tuple representing the colours of the vertices (res[i] is the colour of vertex i)
    :return: True if all vertices have the same colour, False otherwise.
    """
    return len(set(vertex_colouring)) == 1


class ExperimentGraph:

    def __init__(self):
        self.G = nx.MultiGraph()  # the graph itself, as a networkx multigraph
        self.number_of_nodes = 0  # number of nodes in the graph
        self.number_of_edges = 0  # number of edges in the graph

    def add_node(self, node: int):
        """
        Adds a node to the graph.
        :param node: ID of the node.
        :return: None
        """
        self.G.add_node(node)
        self.number_of_nodes += 1

    def add_nodes_from(self, nodes):
        """
        Adds multiple nodes to the graph.
        :param nodes: list of integers, representing the nodes to add.
        :return: None
        """
        self.G.add_nodes_from(nodes)
        self.number_of_nodes += len(nodes)

    def add_edge(self, u: int, v: int, u_colour, v_colour, c_weight: complex):
        """
        Adds an edge to the graph.
        :param u: first vertex of the edge
        :param v: second vertex of the edge
        :param u_colour: colour on the u-side of the edge
        :param v_colour: colour on the v-side of the edge
        :param c_weight: a complex number representing the weight of the edge
        :return: None
        """
        assert u in self.G.nodes and v in self.G.nodes, "Nodes must be in the graph before adding an edge."
        assert u != v, "Self-loops are not allowed."
        self.G.add_edge(u, v, colour={u: u_colour, v: v_colour}, c_weight=c_weight)
        self.number_of_edges += 1

    def get_perfect_matchings(self, graph: nx.MultiGraph = None):
        """
        Returns a list of all perfect matchings of the graph.
        :return: list of sets of tuples (u, v) representing the edges of the perfect matchings.
        """
        if graph is None:   # First time this function is called
            graph = self.G

        if graph.number_of_nodes() == 2:
            return [{edge} for edge in graph.edges]

        perfect_matchings = []
        for u, v in itertools.combinations(graph.nodes, 2):
            if graph.has_edge(u, v):
                working_graph = deepcopy(graph)
                working_graph.remove_node(u)
                working_graph.remove_node(v)

                pms_of_subgraph = self.get_perfect_matchings(working_graph)
                for edge in graph[u][v]:
                    for pm in pms_of_subgraph:
                        new_pm = pm.union({(u, v, edge)})
                        if new_pm not in perfect_matchings:
                            perfect_matchings.append(pm.union({(u, v, edge)}))

        return perfect_matchings

    def get_feasible_vertex_colourings_weights(self):
        """
        Returns a dictionary of all feasible vertex colourings and their weights.
        :return: dict
        """
        feasible_vertex_colourings = dict()
        for perfect_matching in self.get_perfect_matchings():
            vertex_colouring = self.get_induced_vertex_colouring(perfect_matching)
            if vertex_colouring in feasible_vertex_colourings:
                feasible_vertex_colourings[vertex_colouring] += self.weight(perfect_matching)
            else:
                feasible_vertex_colourings[vertex_colouring] = self.weight(perfect_matching)
        return feasible_vertex_colourings

    def get_induced_vertex_colouring(self, perfect_matching):
        """
        Given a perfect matching, returns the vertex colouring that it induces.
        :param perfect_matching: list of tuples (u, v) representing the edges of the perfect matching
        :return: tuple representing the colours of the vertices (res[i] is the colour of vertex i)
        """
        induced_vertex_colouring = [0] * self.G.number_of_nodes()
        for (u, v, edge_id) in perfect_matching:
            induced_vertex_colouring[u] = self.G[u][v][edge_id]["colour"][u]
            induced_vertex_colouring[v] = self.G[u][v][edge_id]["colour"][v]
        return tuple(induced_vertex_colouring)

    def weight(self, edge_set):
        """
        Given a set of edges, returns the weight of the perfect matching that they represent.
        :param edge_set: list of tuples (u, v) representing the edges of the perfect matching.
        :return: a complex number representing the weight of the perfect matching.
        """
        if len(edge_set) == 0:
            return 0  # An empty set of edges has weight 0

        current_weight = 1
        for (u, v, edge_id) in edge_set:
            current_weight *= self.G[u][v][edge_id]["c_weight"]
        return current_weight

    def is_perfectly_monochromatic(self):
        """
        Returns True if the graph is perfectly monochromatic, False otherwise.
        :return: bool
        """
        feasible_vertex_colourings_weights = self.get_feasible_vertex_colourings_weights()
        for vertex_colouring in feasible_vertex_colourings_weights:
            if is_monochromatic(vertex_colouring):
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
            if is_monochromatic(vertex_colouring):
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

        for u, v, edge_id in self.G.edges:
            number_of_edges = len(self.G[u][v])
            intermediate_position = get_coordinates_of_intermediate_point(node_positions[u], node_positions[v], edge_id, number_of_edges)
            u_colour = self.G[u][v][edge_id]["colour"][u]
            v_colour = self.G[u][v][edge_id]["colour"][v]
            c_weight = self.G[u][v][edge_id]["c_weight"]
            latex += "\\draw [style=thin, color=" + str(u_colour) + "] (" + str(node_positions[u][0]) + "," + str(node_positions[u][1]) + ") to (" + str(intermediate_position[0]) + "," + str(intermediate_position[1]) + ");\n"
            latex += "\\draw [style=thin, color=" + str(v_colour) + "] (" + str(intermediate_position[0]) + "," + str(intermediate_position[1]) + ") to (" + str(node_positions[v][0]) + "," + str(node_positions[v][1]) + ");\n"
            latex += "\\node [style=circle, draw=none] at (" + str(intermediate_position[0]) + "," + str(intermediate_position[1]) + ") {" + weight_to_string(c_weight) + "};\n"

        for node in self.G.nodes:
            x, y = node_positions[node]
            latex += "\\node [style=circle, fill=white, draw=black] (" + str(node) + ") at (" + str(x) + "," + str(y) + ") {" + str(node) + "};\n"

        latex += "\\end{tikzpicture}\n"
        latex += "\\end{document}"
        print("we finished creating the latex")
        return latex

    def to_pdf(self, folder: str, filename:str):
        """
        Generates a PDF file with the LaTeX representation of the graph.
        :param folder: folder where the PDF file will be stored.
        :param filename: name of the file to generate (without the .tex).
        :return: None
        """
        path = folder + "/" + filename
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

        assert self.number_of_nodes > 0, "The graph must have at least one node."
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


def weight_to_string(weight: complex):
    """
    Given a complex number, returns a string representation of it.
    :param weight: complex number to convert.
    :return: str
    """
    # First, we get rid of the unnecessary trailing zeros
    real_part = weight.real
    imaginary_part = weight.imag
    if real_part == int(real_part):
        real_part = int(real_part)
    if imaginary_part == int(imaginary_part):
        imaginary_part = int(imaginary_part)

    if weight.imag == 0:    # Pure real number
        return str(real_part)
    if weight.real == 0:    # Pure imaginary number
        if weight.imag == 1:
            return "i"
        if weight.imag == -1:
            return "-i"
        return str(imaginary_part) + "i"
    # Else, we have a complex number
    if imaginary_part > 0:  # The imaginary part is positive
        if imaginary_part == 1:
            return str(real_part) + "+i"
        return str(real_part) + "+" + str(imaginary_part) + "i"
    # Else, the imaginary part is negative
    if imaginary_part == -1:
        return str(real_part) + "-i"
    return str(real_part) + "-" + str(-imaginary_part) + "i"

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


def powerset(iterable, min_size=0):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(min_size, len(s) + 1))


def generate_all_graphs(number_of_nodes: int,
                        number_of_colours: int,
                        real_numbers: list,
                        imaginary_numbers: list,
                        results_folder: str,
                        min_matching_index: int):
    """
    Draws all possible graphs with the given parameters.
    :param number_of_nodes: number of nodes in the graph
    :param number_of_colours: determines the number of colours to use for the edges
    :param real_numbers: list of real numbers to use as the real part of the weights
    :param imaginary_numbers: list of real numbers to use as the imaginary part of the weights
    :param results_folder: folder where the results will be stored
    :param min_matching_index: minimum matching index to consider
    :return:
    """
    possible_colours = ["red", "green", "blue", "orange", "purple", "yellow", "black", "pink", "brown"][:number_of_colours]
    possible_edge_positions = list(itertools.combinations(range(number_of_nodes), 2))
    possible_edge_bicolours = list(itertools.combinations(possible_colours, 2))
    possible_complex_weights = list(itertools.product(real_numbers, imaginary_numbers))
    possible_edges = list(itertools.product(possible_edge_positions, possible_edge_bicolours, possible_complex_weights))

    perfectly_monochromatic_graphs_counter = 0
    graph_counter = 0

    for edge_set in powerset(possible_edges, min_size=number_of_nodes // 2):
        graph_counter += 1
        print(f"Graph {graph_counter} / around {2 ** len(possible_edges)}")
        exp_graph = ExperimentGraph()
        exp_graph.add_nodes_from(range(number_of_nodes))
        for edge in edge_set:
            u, v = edge[0]
            u_colour, v_colour = edge[1]
            c_weight = edge[2]
            exp_graph.add_edge(u, v, u_colour, v_colour, c_weight)
        if exp_graph.get_weighted_matching_index() == 2:
            perfectly_monochromatic_graphs_counter += 1
            exp_graph.to_pdf(results_folder, "graph_" + str(perfectly_monochromatic_graphs_counter))
        del exp_graph


if __name__ == "__main__":
    # n = int(input("Enter the size of the graph (number of nodes): "))

    """
    G = ExperimentGraph()
    G.add_nodes_from((0, 1, 2, 3, 4, 5))
    # Green edges
    G.add_edge(0, 1, "green", "green", 1 + 0j)
    G.add_edge(2, 3, "green", "green", 1 + 0j)
    G.add_edge(4, 5, "green", "green", 1 + 0j)
    G.add_edge(2, 5, "green", "green", 1 + 0j)
    # Red edges
    G.add_edge(1, 2, "red", "red", 1 + 0j)
    G.add_edge(1, 2, "green", "red", 1 + 1j)
    G.add_edge(3, 4, "red", "red", 1 + 0j)
    G.add_edge(5, 0, "red", "red", 1 + 0j)
    # Mixed edges
    G.add_edge(2, 4, "green", "red", 0 + 1j)
    G.add_edge(2, 4, "red", "green", -1 - 1j)
    G.add_edge(2, 4, "blue", "orange", -1 - 1j)
    G.add_edge(5, 3, "green", "red", 0 + 1j)
    
    G.to_pdf("results", "graph_1")
    """

    generate_all_graphs(6,
                        2,
                        [-1, 0, 1],
                        [-1, 0, 1],
                        "results/n_6_c_2",
                        2)
