# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 21:34:06 2020

@author: Sulla
"""

import os
import itertools
import difflib

import os.path as op
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import distance



def factorial(n):
    if n < 2:
        return 1
    else:
        return n * factorial(n - 1)


def generate_coords(vmin = 0, vmax = 10, size = 2):
    """Return list of random integer coordinates for a point between 
        vmin (0) and vmax (10) in dimension size (2)
    """
    return list(np.random.randint(vmin, vmax, size))


def generate_locs(num = 4, coord_min = 0, coord_max = 10):
    """Return random dict of {location:[coordinates]} 
        where the num (4) locations have coordinates
        between coord_min (0) and coor_max (10)
    """
    
    if num > (coord_max - coord_min):
        raise ValueError('Too many locations requested for the coordnates given')
    
    locs = {}
    for loc in range(num):
        coords = []
        #Avoid adding the same coordinates multiple times
        while (coords == []) or (coords in locs.values()):
            coords = generate_coords(coord_min, coord_max)
        
        locs[loc] = coords
    
    return locs


def plot_locs(locs, title = None):
    """Make plot with the locactions in a loc dict
    """
    
    fig, ax = plt.subplots()
    for loc in range(len(locs)):
        plt.scatter(*locs[loc])
        ax.annotate(loc, locs[loc])
        
    if not title == None:
        plt.title(title)


def plot_path(locs, edge_list, title = None):
    """Make plot with the locactions in a loc dict and add the path.
        Can also pass along partial solutions.
    """
    
    plot_locs(locs, title)
        
    for ii, edge in enumerate(edge_list):
        if ii == 0:
            plt.annotate('S', locs[edge[0]], textcoords="offset points")
        
        edge_locs = locs[edge[0]], locs[edge[1]]
        xs = [loc[0] for loc in edge_locs]
        ys = [loc[1] for loc in edge_locs]
        plt.plot(xs, ys, 'k-')
    
    plt.figure()


def build_plot_path(locs, edge_list):
    """Plot the path one line at a time
    """
    
    for ii in range(len(edge_list)):
        plot_path(locs, edge_list[:ii+1])


def get_distances(locs):
    """Return dict {[loc_i, loc_j]: distance} for every location in locs
    """
    size = len(locs)
    dists = {}
    
    for idx in range(size):
        if idx == (size - 1):
            continue
        else:
            for idx2 in range(idx + 1, size):
                dists[(idx, idx2)] = np.round(distance.euclidean(locs[idx], locs[idx2]), 3)
    
    return dists


def create_graph(dists):
    """Return networkx graph version of a distance dict
    """
    graph = nx.Graph()
    
    for key in dists.keys():
        weight = dists[key]
        graph.add_edge(key[0], key[1], weight = weight)
    
    return graph


def draw_graph(graph, dists):
    """Draw networkx graph object using spring layout and show distances
        as edge weights
    """
    
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos)
    nx.draw_networkx_edge_labels(graph, pos, dists)
    plt.show()


def get_cycle(nodes):
    
    nodes = list(nodes).copy()
    np.random.shuffle(nodes)
    
    return convert_to_edges(nodes)


def get_all_cycles(nodes):
    """Return a list of the permutation of a list of locations
    
    nodes: list
    """
    
    return list(itertools.permutations(nodes))


def get_all_tsp_solutions(tsp, start):
    """Return list of the a given tsp's solutions sample space.
        These are all cycles/permutations beginning with start.
    
    tsp: nx graph object
    
    start: node type of tsp
    """
    
    nodes = list(tsp.nodes())
    
    #Generate list of all cycles/permutations of all locs minus the start loc
    nodes.remove(start)
    all_cycles = get_all_cycles(nodes)
    
    #Avoid double counting
    #TSP solutions start and end with start.
    #They are the same when reversed.
    cycles, added = [], []
    for cycle in all_cycles:
        #Prevent reversed cycle being added to list
        rev = cycle[::-1]
        if rev not in added:
            added.append(cycle)
            
            #Make sure tsp cycle starts at given loc
            full_cycle = tuple([start] + list(cycle))
            cycles.append(full_cycle)
    
    return cycles


def get_solutions_dict(solutions, edge_dists):
    """Return dictionary with information for each of the solutions, which are arbitrarily indexed.
    
    solutions: list
        List of solutions to a given TSP. Each solution must represent the path
        as a list that ends with the location prior to going back to the start.
    edge_dists: dict
        Dict of the edges of a given TSP that returns the distance between nodes
    
    RETURN:
        dict
            solution: list of solution path
            dist: total travel distance of solution
            sim: similarity to a select main solution, initialized to -100
                Must use with redo_sim() to make this key useful.
    """
    
    solutions_dict = {}    
    for idx, solution in enumerate(solutions):
        solutions_dict[idx] = {'solution': solution} #Solution path key
        
        #Solution travel distance
        solutions_dict[idx]['dist'] = sum_dists(convert_to_edges(solution), edge_dists)
        
        #
#        reg_sim = difflib.SequenceMatcher(None, main_solution, solution).ratio()
#        rev_tsp_solution = [main_solution] + list(solution[1:][::-1])
#        rev_sim = difflib.SequenceMatcher(None, main_solution, rev_tsp_solution).ratio()
        
        #Initialize similarity measure to very low for all solutions
        solutions_dict[idx]['sim'] = -100
    
    return solutions_dict


def convert_to_edges(path_as_list):
    """Convert a list representation of a path into 
        a list of the edges that make up that path
    
    path_as_list: list
        For TSP solutions, the given path must not end back at start.
    """
    
    size = len(path_as_list)
    edge_list = []
    for idx in range(size):
        if idx == size - 1:
            edge_list.append((path_as_list[idx], path_as_list[0])) #Add edge back to start
        else:
            edge_list.append((path_as_list[idx], path_as_list[idx+1]))
    
    return edge_list


def sum_dists(solution, edge_dists):
    """Return the total travel distance of a given solution
    
    solution: list
        List of edges that make up the path
    
    edge_dists: dict
        {(edge): distance}
    """
    
    solution_dists = []
    
    for edge in solution:
        if edge in edge_dists.keys():
            solution_dists.append(edge_dists[edge])
        else:
            solution_dists.append(edge_dists[edge[::-1]])
    
    return sum(solution_dists)


def get_sols_idx_sorted(solutions_dict, sort_by = 'dist'):
    """Return a list of the solutions' index in solutions_dict sorted by sort_by and
        a list of those values sorted
    
    solutions_dict: dict
        Dictionary with information for each of the solutions, which are arbitrarily indexed
    sort_by: object ('dist')
        Any python object that can be a dict key. Must be a key of the sub-dicts in solutions_dict
    """
    
    solutions_idx, solutions_dists = [], []
    for idx in solutions_dict.keys():
        solutions_idx.append(idx)
        solutions_dists.append(solutions_dict[idx][sort_by])
    
    solutions_idx_vals = {key:val for key, val in zip(solutions_idx, solutions_dists)}
    solutions_idx_sorted_by_vals = [item[0]
                                    for item in sorted(solutions_idx_vals.items(), key = lambda item: item[1])]
    sorted_by_vals = [item[1]
                        for item in sorted(solutions_idx_vals.items(), key = lambda item: item[1])]

    return solutions_idx_sorted_by_vals, sorted_by_vals


def update_sim(solutions_dict, comparison_solution):
    """Update the similarity measures for all the solutions/path in the solutions dict
        using the comparison_solution. Given the symmetry of the solutions to TSPs,
        similarity is defined as the max value of two comparisons:
            (1): the regular solution
            (2): the solution with all the locations in the path reversed except the start
    
    Similarity if based on difflib.SequenceMatcher:
         "This does not yield minimal edit sequences, but does tend to yield 
         matches that "look right" to people."
    
    solutions_dict: dict
        Dict of information for all solutions. Must have a 'sim' key for all solutions.
    comparison_solution: list
    """
    
    solutions = [solutions_dict[key]['solution'] for key in solutions_dict.keys()]
    
    for idx, solution in enumerate(solutions):
        reg_sim = difflib.SequenceMatcher(None, comparison_solution, solution).ratio()
        rev_tsp_sol = [solution[0]] + list(solution[1:][::-1])
        
        rev_sim = difflib.SequenceMatcher(None, comparison_solution, rev_tsp_sol).ratio()
        solutions_dict[idx]['sim'] = max(reg_sim, rev_sim)


def lists_to_df(lists, col_names = None):
    """Conver list of values into df using col_names
    """
    
    if col_names == None:
        col_names = [str(num) for num in range(len(lists))]
    
    return pd.DataFrame.from_dict({col:vals for vals, col in zip(lists, col_names)})


def get_prox_dists_for_node(node, path_idx_to_prox, paths_to_dist):
    """Return dict linking distances of all paths arranged by the proximal location
        of the given node
    
    node: object
        Any object that is an item in the path lists
    path_idx_to_prox: dict
        Dict mapping from path list index to proximity
    paths_to_dist: dict
        Dict mapping path lists to total distance of the path
    """
    
    node_proxs_to_dists = {}
    for path in paths_to_dist.keys():
        node_loc_in_path = path.index(node)
        node_proximity_to_start = path_idx_to_prox[node_loc_in_path]
        dist = paths_to_dist[path]
        
        if node_proximity_to_start not in node_proxs_to_dists.keys():
            node_proxs_to_dists[node_proximity_to_start] = []
        
        node_proxs_to_dists[node_proximity_to_start].append(dist)
    
    return node_proxs_to_dists



def get_best_next_node(num_locs, start_loc, path_idx_to_prox, paths_to_dist, skip_list = [], prox = 1, path_sample_size = 3000, samples = 10):
    """
    """
    
    nodes_to_next_prox_means = {}
    skip_list.append(start_loc)
    
    for node in range(num_locs):
        if node in skip_list:
            continue
        
        nodes_to_next_prox_means[node] = []
        node_info = get_prox_dists_for_node(node, path_idx_to_prox, paths_to_dist)
        
        for sample_num in range(samples):
            
            #sample from paths with given proximity for node
            node_promixity_sample_mean = np.mean(np.random.choice(node_info[prox], path_sample_size, replace = False))
            
            #Update running average of sample means
            if sample_num == 1:
                nodes_to_next_prox_means[node] = node_promixity_sample_mean
            else:
                old_mean = nodes_to_next_prox_means[node]
                nodes_to_next_prox_means[node] = old_mean + ((node_promixity_sample_mean - old_mean) / (sample_num + 1))
    
    return sorted(nodes_to_next_prox_means.items(), key = lambda item: item[1])[0][0]
