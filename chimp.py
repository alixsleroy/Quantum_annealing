
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

from chimp_data import chimp_affiliative, chimp_agonistic, chimp_grooming


from matplotlib import pyplot, patches

import numpy as np
from signet.cluster import Cluster
from scipy.sparse import csc_matrix


def cost(G, G0, G1):
    # compute cost
    cost1 = 0
    cost2 = 0
    for i,j in G.edges:
        if (i in G0 and j in G0) or (i in G1 and j in G1):
            cost1 -= 1
        elif (i in G0 and j in G1):
            cost1 += -2
        elif (i in G1 and j in G0):
            cost1 += 2
        else:
            print('Assigning error!')

    for i,j in G.edges:
        if (i in G0 and j in G0) or (i in G1 and j in G1):
            cost2 -= 1
        elif (i in G0 and j in G1):
            cost2 += 2
        elif (i in G1 and j in G0):
            cost2 += -2
        else:
            print('Assigning error!')
            

    cost = min(cost1, cost2)
    return cost
    
def new_mapping(A):
    ''' Compute the new mapping f(A) for the adjacency matrix A of the network.
    '''
    # Compute the SVD decompostion.
    u, s, v = np.linalg.svd(A, full_matrices=True)
    # Compute f(A).
    fA = u @ np.diag(np.cosh(s)) @ u.T - u @ np.diag(np.sinh(s)) @ v
    return fA

def spectual(G):

    A = nx.adjacency_matrix(G)

    fA = new_mapping(A.todense())

    fAT = new_mapping(A.todense().T)

    M = fA + fAT

    M_csc = csc_matrix(M)
    M_bar = abs(M_csc)
    M_p = (M_csc + M_bar) / 2
    M_n = - (M_csc - M_bar) / 2

    M_p.eliminate_zeros()
    M_n.eliminate_zeros()

    C = Cluster((M_p, M_n))

    pcapreds = C.SPONGE_sym(2)

    # Process the result
    G0 = []
    G1 = []
    n = G.number_of_nodes()
    G0 = [list(G.nodes)[i] for i in range(n) if pcapreds[i] == 0]
    G1 = [list(G.nodes)[i] for i in range(n) if pcapreds[i] == 1]

    return (cost(G, G0, G1), G0, G1)


## ------- Create the graph -------
#np.random.seed(123)
G = chimp_agonistic()

print(G)

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
print("\n")
print("Quantum Annealing")
print('-' * 60)
print('{:>15s}{:>15s}{:^15s}{:^15s}'.format('Set 0','Set 1','Energy','Cut Size'))
print('-' * 60)
for sample, E in response.data(fields=['sample','energy']):
    S0 = [k for k,v in sample.items() if v == 0]
    S1 = [k for k,v in sample.items() if v == 1]
    #Enew = int(E) + G.number_of_edges() 
    Enew = cost(G, S0, S1)
    #Enew = int(E) + count_nodes(S0,S1,G) 
    # when you use stochastic bloc
    # you get the issue but not with exact compact graph 
    # I am sure why
    # count_nodes(set1,set2,G) is  a new function in synthetic_data that counts the number 
    # of nodes going from 1 to 2. 
    print('{:>15s}{:>15s}{:^15s}{:^15s}'.format(str(S0),str(S1),str(Enew),str(int(-1*Enew))))

energylist = []
for sample, E in response.data(fields=['sample','energy']):
    energylist.append(E)

#print(min(energylist))


