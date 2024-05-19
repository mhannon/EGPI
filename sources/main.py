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


def random_experiment_graphs_research(number_of_trials: int,
                                      number_of_nodes: int,
                                      colours: list,
                                      weights: list,
                                      complexity_bounds: list,
                                      results_folder: str):
    """
    Draws a random graph with the given parameters.
    :param number_of_trials: number of graphs to draw
    :param number_of_nodes: number of nodes in the graph
    :param colours: list of colours to use for the edges
    :param weights: list of complex numbers to use as the weights
    :param complexity_bounds: list of the complexity bounds to use
    :param results_folder: folder where the results will be stored
    :return: None
    """
    number_of_discovered_graphs = 0
    used_complexity_bound = 0

    for i in range(number_of_trials):  # Generate m random graphs

        # Display advancement in terminal for convenience
        display_advancement(i, number_of_trials, number_of_discovered_graphs)
        # Loop through the complexity bounds
        used_complexity_bound = (used_complexity_bound + 1) % len(complexity_bounds)
        # Generate a random candidate graph with our parameters
        candidate_graph = generate_random_candidate_graph(number_of_nodes, colours, weights,
                                                          complexity_bounds[used_complexity_bound])
        # Get the feasible vertex colourings of the generated graph
        feasible_vertex_colourings_weights = candidate_graph.get_feasible_vertex_colourings_weights()
        is_monochromatic = not any(len(set(vertex_colouring)) > 1
                                   for vertex_colouring in feasible_vertex_colourings_weights)
        # Save the results if they are of interest
        if candidate_graph.get_weighted_matching_index() == len(colours) \
                and not is_monochromatic:
            number_of_discovered_graphs += 1
            save_candidate_graph(results_folder, candidate_graph, number_of_nodes, len(colours), len(weights),
                                 complexity_bounds[used_complexity_bound])


def display_advancement(current_trials, total_trials, number_of_discovered_graphs):
    """
    Displays the advancement of the generation of the graphs in the terminal.
    :param current_trials: number of graphs generated so far
    :param total_trials: total number of graphs to generate
    :param number_of_discovered_graphs: number of graphs that have been found
    :return: None
    """
    if current_trials % 10000 == 0:
        print("Generating graph " + str(current_trials) + " / " + str(total_trials) + "... (" + str(
            current_trials * 100 // total_trials) + "%) - " + str(number_of_discovered_graphs) + " graphs found.")


def generate_random_candidate_graph(number_of_nodes, colours, weights, complexity_bound):
    """
    Generates a random candidate graph with the given parameters.
    :param number_of_nodes: of the graph.
    :param colours: to use for the edges.
    :param weights: to use for the edges.
    :param complexity_bound: number of attempts to add a random perfect matching.
    :return: the generated experiment graph.
    """
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
    for _ in range(complexity_bound):
        add_random_perfect_matching(exp_graph, weights, colours)
    return exp_graph


def add_random_perfect_matching(exp_graph: ExperimentGraph, weights: list, colours: list):
    """
    Adds random edges to the graph that belong to at least one perfect matching.
    :param exp_graph: graph to add the edges to.
    :param weights: to use for the edges.
    :param colours: to use for the edges.
    :return:
    """
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

        for bicolour in present_bicolours:
            possible_bicolours.remove(bicolour)
        if len(possible_bicolours) > 0:
            u_colour, v_colour = random.choice(possible_bicolours)

            # Get the weight of the potential new edge
            c_weight = random.choice(weights)

            choice = random.choice(range(len(exp_graph.edges(u, v)) + 1))
            if choice == 0:  # Add a new edge
                exp_graph.add_edge(u, v, u_colour, v_colour, c_weight)


def save_candidate_graph(results_folder, candidate_graph, number_of_nodes, number_of_colours, number_of_weights,
                         complexity_bound):
    """
    Saves the candidate graph in the results folder.
    :param results_folder: name of the root folder where the results will be stored
    :param candidate_graph: experiment graph to save
    :param number_of_nodes: of the graph
    :param number_of_colours: of the edges of the graph
    :param number_of_weights: of the edges of the graph
    :param complexity_bound: used to generate the graph
    :return: None
    """
    folder_name = (results_folder + "/" +
                   str(number_of_nodes) + "_nodes/" +
                   str(number_of_colours) + "_colours/" +
                   str(number_of_weights) + "_weights/" +
                   str(complexity_bound) + "_complexity")
    graph_name = "graph_" + str(number_of_saved_graph_in(folder_name) + 1)
    candidate_graph.to_pdf(folder_name, graph_name)
    candidate_graph.to_json(folder_name, graph_name)


def number_of_saved_graph_in(folder: str):
    """
    Finds the number of graphs in a given folder.
    :param folder: folder to look for the graphs in
    :return: number of graphs in the folder
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    files = os.listdir(folder)
    return len([file for file in files if file.endswith(".json")])


"""
def generate_all_graphs(number_of_nodes: int,
                        colours: list,
                        weights: list,
                        results_folder: str,
                        min_matching_index: int):
    
    Draws all possible graphs with the given parameters.
    :param number_of_nodes: number of nodes in the graph
    :param number_of_colours: determines the number of colours to use for the edges
    :param real_numbers: list of real numbers to use as the real part of the weights
    :param imaginary_numbers: list of real numbers to use as the imaginary part of the weights
    :param results_folder: folder where the results will be stored
    :param min_matching_index: minimum matching index to consider
    :return:
    
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
"""


def main():
    """
    Main function of the program. Asks the user for the parameters of the random experiment graphs
    research process and performs it.
    :return: None
    """
    number_of_graphs = int(input(
        "Enter the number of graphs to generate: "))
    number_of_nodes = int(input(
        "Enter the size of the randomly generated graphs (number of nodes): "))
    colours = input(
        "Enter the colours to use for the edges (separated by a space): ").split()
    weights = [complex(weight) for weight in input(
        "Enter the weights to use for the edges (separated by a space, in the form a+bj): ").split()]
    complexity_bounds = [int(x) for x in input(
        "Enter the complexity bound(s) to use (separated by a space): ").split()]

    random_experiment_graphs_research(number_of_graphs,
                                      number_of_nodes,
                                      colours,
                                      weights,
                                      complexity_bounds,
                                      "results")


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
