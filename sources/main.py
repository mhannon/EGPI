"""
Author: Merlin Hannon
Date: 2024-05-07
"""


import itertools

from sources.ExperimentGraph import ExperimentGraph


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
        if exp_graph.get_weighted_matching_index() >= min_matching_index:
            perfectly_monochromatic_graphs_counter += 1
            exp_graph.to_pdf(results_folder, "graph_" + str(perfectly_monochromatic_graphs_counter))
        del exp_graph


if __name__ == "__main__":
    # n = int(input("Enter the size of the graph (number of nodes): "))

    G = ExperimentGraph()
    G.add_nodes_from((0, 1, 2, 3, 4, 5))
    # Green edges
    G.add_edge(0, 1, "green", "green", 1 + 0j)
    G.add_edge(2, 3, "green", "green", 1 + 0j)
    G.add_edge(4, 5, "green", "green", 1 + 0j)
    G.add_edge(2, 5, "green", "green", 1 + 0j)
    # Red edges
    G.add_edge(1, 2, "red", "red", 1 + 0j)
    G.add_edge(3, 4, "red", "red", 1 + 0j)
    G.add_edge(5, 0, "red", "red", 1 + 0j)
    # Mixed edges
    G.add_edge(2, 4, "green", "red", 0 + 1j)
    G.add_edge(3, 5, "red", "green", 0 + 1j)

    G.to_pdf("../results", "graph_1")

    for pm in G.get_perfect_matchings():
        print("PM: " + str(pm))     # TODO : every perfect matching seems to be counted twice
    print(G.get_weighted_matching_index())
    print(G.is_perfectly_monochromatic())


"""
    generate_all_graphs(6,
                        2,
                        [-1, 0, 1],
                        [-1, 0, 1],
                        "results/n_6_c_2",
                        2)
"""
