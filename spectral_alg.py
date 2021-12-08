
from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import networkx as nx
import numpy as np

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

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
            

    cost = max(cost1, cost2)
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

    #pcapreds = C.SPONGE_sym(2)
    pcapreds = C.SPONGE(2)

    # Process the result
    G0 = []
    G1 = []
    n = G.number_of_nodes()
    G0 = [list(G.nodes)[i] for i in range(n) if pcapreds[i] == 0]
    G1 = [list(G.nodes)[i] for i in range(n) if pcapreds[i] == 1]

    return (cost(G, G0, G1), G0, G1)
