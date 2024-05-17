"""
Author: Merlin Hannon
Date: 2024-05-07
"""

import itertools
import os
import random
from sources.ExperimentGraph import ExperimentGraph


def powerset(iterable, min_size=0):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(min_size, len(s) + 1))


def add_random_perfect_matching(exp_graph: ExperimentGraph, weights: list, colours: list):
    # Positions of the edges of the perfect matching
    random_vertex_order = random.sample(range(exp_graph.number_of_nodes), exp_graph.number_of_nodes)

    # Add edges that will be part of perfect matchings at this position
    for edge_index in range(exp_graph.number_of_nodes // 2):
        # Get the potential new edge's position
        u, v = random_vertex_order[2 * edge_index], random_vertex_order[2 * edge_index + 1]

        # Get the bicolour of the potential new edge
        edges = exp_graph.edges(u, v)
        present_bicolours = [(edge.get_u_colour(), edge.get_v_colour()) for edge in edges]
        possible_bicolours = [(u_colour, v_colour) for u_colour in colours for v_colour in colours]
        # print("Present bicolours : " + str(present_bicolours))
        # print("Possible bicolours : " + str(possible_bicolours))
        for bicolour in present_bicolours:
            possible_bicolours.remove(bicolour)
        if len(possible_bicolours) > 0:
            u_colour, v_colour = random.choice(possible_bicolours)

            # Get the weight of the potential new edge
            c_weight = random.choice(weights)

            choice = random.choice(range(len(exp_graph.edges(u, v)) + 1))
            if choice == 0:  # Add a new edge
                exp_graph.add_edge(u, v, u_colour, v_colour, c_weight)


def generate_m_random_graph(number_of_trials: int,
                            number_of_nodes: int,
                            colours: list,
                            weights: list,
                            results_folder: str,
                            min_matching_index: int):
    """
    Draws a random graph with the given parameters.
    :param number_of_trials: number of graphs to draw
    :param number_of_nodes: number of nodes in the graph
    :param colours: list of colours to use for the edges
    :param weights: list of complex numbers to use as the weights
    :param results_folder: folder where the results will be stored
    :param min_matching_index: minimum matching index to consider
    :return:
    """
    number_of_discovered_graphs = 0
    number_of_added_PMs = 0

    for i in range(number_of_trials):  # Generate m random graphs

        # Display advancement for convenience
        if i % 10000 == 0:
            print("Generating graph " + str(i) + " / " + str(number_of_trials) + "... (" + str(
                i * 100 // number_of_trials) + "%) - " + str(number_of_discovered_graphs) + " graphs found.")

        # Create new empty experiment graph
        exp_graph = ExperimentGraph()
        exp_graph.add_nodes_from(range(number_of_nodes))

        # Generate at least one perfect matching of each colour
        for colour in colours:
            random_node_order = random.sample(range(number_of_nodes), number_of_nodes)
            for edge_index in range(number_of_nodes // 2):
                u, v = random_node_order[2 * edge_index], random_node_order[2 * edge_index + 1]
                c_weight = random.choice(weights)
                exp_graph.add_edge(u, v, colour, colour, c_weight)
                # TODO : add a step to ensure that the weights of all PMs are 1

        # Add random perfect matchings to the already present ones
        number_of_added_PMs = (number_of_added_PMs % 6) + 1  # TODO : change this to a parameter
        for _ in range(number_of_added_PMs):
            add_random_perfect_matching(exp_graph, weights, colours)

        # Get the feasible vertex colourings of the generated graph
        feasible_vertex_colourings_weights = exp_graph.get_feasible_vertex_colourings_weights()
        is_monochromatic = not any(len(set(vertex_colouring)) > 1
                                   for vertex_colouring in feasible_vertex_colourings_weights)

        # Draw the graph
        if exp_graph.get_weighted_matching_index() >= min_matching_index \
                and not is_monochromatic:
            number_of_discovered_graphs += 1
            exp_graph.to_pdf(results_folder, "graph_" + str(number_of_discovered_graphs))
            exp_graph.to_json(results_folder + "/graph_" + str(number_of_discovered_graphs))


def generate_all_graphs(number_of_nodes: int,
                        colours: list,
                        weights: list,
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
    possible_edge_positions = list(itertools.combinations(range(number_of_nodes), 2))
    possible_edges = list(itertools.product(possible_edge_positions, colours, weights))

    perfectly_monochromatic_graphs_counter = 0
    graph_counter = 0

    for edge_set in powerset(possible_edges, min_size=number_of_nodes // 2):
        graph_counter += 1
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


def main():
    number_of_graphs = int(input("Enter the number of graphs to generate: "))
    n = int(input("Enter the size of the randomly generated graphs (number of nodes): "))
    colours = input("Enter the colours to use for the edges (separated by a space): ").split()
    weights_as_strings = input(
        "Enter the weights to use for the edges (separated by a space, in the form a+bj): ").split()
    weights = [complex(weight) for weight in weights_as_strings]
    min_matching_index = int(input("Enter the minimum matching index to consider: "))
    results_folder = "../results/" + \
                     str(n) + "_nodes/" + \
                     str(len(colours)) + "_colours/" + \
                     str(len(weights)) + "_weights/" + \
                     str(min_matching_index) + "_matching_index"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    generate_m_random_graph(number_of_graphs,
                            n,
                            colours,
                            weights,
                            results_folder,
                            min_matching_index)


if __name__ == "__main__":
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

    main()
