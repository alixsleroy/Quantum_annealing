# Copyright 2019 D-Wave Systems, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------ Import necessary packages ----
from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import networkx as nx
import numpy as np

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

# from graph_creation import compexact_bipartipe_graph, stochastic_block_model
from greedy_algAlix import greedy_algorithm
from synthetic_data import stochastic_block_model
from synthetic_data import compexact_bipartipe_graph


## ------- Create the graph -------
#np.random.seed(123)
G = stochastic_block_model(10)

## ********************* TRADITIONAL APPROACH *********************
## ------- Resolve using Spectral algorithm -------
print("Spectral Algorithm")
print('-' * 60)
print('{:>15s}{:>15s}{:^15s}'.format('Set 0','Set 1','Cut Size'))
print('-' * 60)
## still need to do



## ********************* QUANTUM APPROACH *********************
## ------- Set up our QUBO dictionary -------
Q = defaultdict(int)

# # Update Q matrix for every edge in the graph
for i, j in G.edges:
    Q[(i,i)]+= 1
    Q[(j,j)]-= 3
    Q[(i,j)]+= 2

# # ------- Run our QUBO on the QPU -------
# Set up QPU parameters
chainstrength = 8
numruns = 10

# # Run the QUBO on the solver from your config file
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_qubo(Q,
                               chain_strength=chainstrength,
                               num_reads=numruns,
                               label='Example - Maximum Cut')

# # ------- Print results to user -------
print("\n")
print("Quantum Annealing")
print('-' * 60)
print('{:>15s}{:>15s}{:^15s}{:^15s}'.format('Set 0','Set 1','Energy','Cut Size'))
print('-' * 60)
for sample, E in response.data(fields=['sample','energy']):
    S0 = [k for k,v in sample.items() if v == 0]
    S1 = [k for k,v in sample.items() if v == 1]
    print('{:>15s}{:>15s}{:^15s}{:^15s}'.format(str(S0),str(S1),str(E),str(int(-1*E))))

energylist = []
for sample, E in response.data(fields=['sample','energy']):
    energylist.append(E)

print(min(energylist))

# print(response.data(fields=['energy']))



# # ------- Display results to user -------
# # Grab best result
# # Note: "best" result is the result with the lowest energy
# # Note2: the look up table (lut) is a dictionary, where the key is the node index
# #   and the value is the set label. For example, lut[5] = 1, indicates that
# #   node 5 is in set 1 (S1).
# lut = response.first.sample
# print(lut)
# # Interpret best result in terms of nodes and edges
# S0 = [node for node in G.nodes if not lut[node]]
# S1 = [node for node in G.nodes if lut[node]]
# cut_edges = [(u, v) for u, v in G.edges if lut[u]!=lut[v]]
# uncut_edges = [(u, v) for u, v in G.edges if lut[u]==lut[v]]

# # Display best result
# pos = nx.spring_layout(G)
# nx.draw_networkx_nodes(G, pos, nodelist=S0, node_color='r')
# nx.draw_networkx_nodes(G, pos, nodelist=S1, node_color='c')
# nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=3)
# nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, style='solid', width=3)
# nx.draw_networkx_labels(G, pos)

# filename = "maxcut_plot.png"
# plt.savefig(filename, bbox_inches='tight')
# print("\nYour plot is saved to {}".format(filename))
