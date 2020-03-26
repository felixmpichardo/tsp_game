# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:42:45 2020

@author: Sulla
"""


import os
import itertools
import difflib
import pickle
import math

import os.path as op
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import distance

from TSP_functions import *



###
#
#   START
#
###

#Generate a random tsp
num_locs = 9
locs = generate_locs(num_locs, coord_max = 20)
edge_dists = get_distances(locs)
tsp = create_graph(edge_dists)

#Visualize tsp
plot_locs(locs, title =  "TSP({}) map".format(num_locs))
plt.figure()
# draw_graph(tsp, edge_dists)

###
#   Brute Force to find Shortest Path
###

#Generate solution space for a given start node
start_loc = np.random.randint(0, len(locs))
tsp_solutions = get_all_tsp_solutions(tsp, start_loc) #solution space
solutions_dict = get_solutions_dict(tsp_solutions, edge_dists) #info on solutions

#Find shortest path
solutions_idx_by_dist, sorted_dists = get_sols_idx_sorted(solutions_dict, 'dist')
solutions_idx_shortest = solutions_idx_by_dist[0]

#Visualize shortest path
shortest_path = solutions_dict[solutions_idx_shortest]['solution']
shortest_path_edges = convert_to_edges(shortest_path)
plot_path(locs, shortest_path_edges, title =  "TSP({}) Shortest Path".format(num_locs))
#build_plot_path(locs, shortest_path_edges)

###
#   Explore similarity of solutions
###

#Update similarity values for solution space with shortest path
update_sim(solutions_dict, shortest_path)

#Sort solutions by similarity
solutions_idx_by_sim, sorted_sims = get_sols_idx_sorted(solutions_dict, 'sim')
solutions_dists_by_sim = [sorted_dists[solutions_idx_by_dist.index(idx)]
                          for idx in solutions_idx_by_sim]

#See least similar path
plot_path(locs, convert_to_edges(solutions_dict[solutions_idx_by_sim[0]]['solution']), 
          title =  "TSP({}) Path Least Similar to Optimal".format(num_locs))

#See relation between similarity and total path distance
#plt.scatter(sorted_sims, solutions_dists_by_sim);plt.figure()

#Save similarity, distance data
dist_by_sim_df = lists_to_df([sorted_sims, solutions_dists_by_sim], ['sims', 'dists'])

save_file = 'dist_by_sim_{}.csv'.format(num_locs)
# dist_by_sim_df.to_csv(save_file, index = None)

#with open("sim_by_dist_{}.pickle".format(num_locs), "wb") as file:
    # pickle.dump(cd, file)

#Explore path with the max distance
max_dist = sorted_dists[-1]
max_dist_idx = solutions_idx_by_dist[-1]
max_dist_sim = solutions_dict[max_dist_idx]['sim']

plot_path(locs, convert_to_edges(solutions_dict[solutions_idx_by_sim[max_dist_idx]]['solution']),
          title =  "TSP({}) Longest Path [Sim: {}]".format(num_locs, np.round(max_dist_sim,3)))


### Using R, I found a significant linear relationship for solution similarity to shortest path
### and solution total distance.

### However, the solution with the max distance is not the least similar.
### It's generally about 50% similar to the optimal solution.

##Proximity of locs to start in optimal solution
#   This is a way to measure the relative location of a node in the path
#   which helps to ignore the symmetry of the solutions.
