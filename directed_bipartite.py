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
from synthetic_data import count_nodes


from matplotlib import pyplot, patches
import numpy as np
from signet.cluster import Cluster
from scipy.sparse import csc_matrix
from spectral_alg import spectual, cost



## ------- Create the graph -------
#np.random.seed(123)
p=0.9
G = stochastic_block_model(6,p/2,1-p/2,p/2,p/2)
#G = compexact_bipartipe_graph(10)

## ********************* TRADITIONAL APPROACH *********************
## ------- Resolve using Spectral algorithm -------
print("Spectral Algorithm")
print('-' * 60)
print('{:>15s}{:>15s}{:^15s}'.format('Set 0','Set 1','Cut Size'))
print('-' * 60)
print(spectual(G))


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
## adjust with the constant
print("\n")
print("Quantum Annealing")
print('-' * 60)
print('{:>15s}{:>15s}{:^15s}{:^15s}'.format('Set 0','Set 1','Energy','Cut Size'))
print('-' * 60)
for sample, E in response.data(fields=['sample','energy']):
    S0 = [k for k,v in sample.items() if v == 0]
    S1 = [k for k,v in sample.items() if v == 1]
    Enew = cost(G, S0, S1)
    # print("QA cost")
    # print('{:>15s}{:>15s}{:^15s}{:^15s}'.format(str(S0),str(S1),str(E),str(int(00))))
    print("JIAAO COST")
    print('{:>15s}{:>15s}{:^15s}{:^15s}'.format(str(S0),str(S1),str(Enew),str(int(00))))




# energylist = []
# for sample, E in response.data(fields=['sample','energy']):
#     energylist.append(E)

# print(min(energylist)+G.number_of_edges())



