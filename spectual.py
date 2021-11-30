# ------ Import necessary packages ----
from collections import defaultdict

# from dwave.system.samplers import DWaveSampler
# from dwave.system.composites import EmbeddingComposite
import networkx as nx

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
from matplotlib import pyplot, patches

import numpy as np
from signet.cluster import Cluster
import numpy as np
from scipy.sparse import csc_matrix

def new_mapping(A):
    ''' Compute the new mapping f(A) for the adjacency matrix A of the network.
    '''
    # Compute the SVD decompostion.
    u, s, v = np.linalg.svd(A, full_matrices=True)

    
    # Compute f(A).
    fA = u @ np.diag(np.cosh(s)) @ u.T - u @ np.diag(np.sinh(s)) @ v
    return fA

    import networkx as nx

def draw_adjacency_matrix(G, node_order=None, partitions=[], colors=[]):
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=node_order)

    #Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5)) # in inches
    pyplot.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none")
    
    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor="none",
                                          edgecolor=color,
                                          linewidth="1"))
            current_idx += len(module)


def spectual(G):
    #draw_adjacency_matrix(G)

    # Compute the adjacency matrix A.
    A = nx.adjacency_matrix(G)
    #print(A.todense())

    # Compute f(A).
    fA = new_mapping(A.todense())
    #print(fA)

    # Create a figure
    #fig, ax = plt.subplots()

    # Plot the heatmap
    #plt.imshow(A.todense(), cmap='cool', interpolation='nearest')
    #plt.colorbar()
    #plt.plot(A.todense())

    #plt.show()

    #plt.imshow(fA, cmap='bwr', interpolation='nearest')
    #plt.colorbar()

    # Set axis properties
    #ax.set_xlabel(r'$x$', fontsize=12)
    #ax.set_ylabel(r'$y$', fontsize=12)

    #plt.show()

    # Compute f(A).
    fAT = new_mapping(A.todense().T)
    #print(fAT)

    # Create a figure
    #fig, ax = plt.subplots()

    # Plot the heatmap
    #plt.imshow(A.todense(), cmap='cool', interpolation='nearest')
    #plt.colorbar()
    #plt.plot(A.todense())

    #plt.show()

    #plt.imshow(fAT, cmap='bwr', interpolation='nearest')
    #plt.colorbar()

    # Set axis properties
    #ax.set_xlabel(r'$x$', fontsize=12)
    #ax.set_ylabel(r'$y$', fontsize=12)

    #plt.show()

    M = fA + fAT

    # Create a figure
    #fig, ax = plt.subplots()

    # Plot the heatmap
    #plt.imshow(A.todense(), cmap='cool', interpolation='nearest')
    #plt.colorbar()
    #plt.plot(A.todense())

    #plt.show()

    #plt.imshow(M, cmap='bwr', interpolation='nearest')
    #plt.colorbar()

    # Set axis properties
    #ax.set_xlabel(r'$x$', fontsize=12)
    #ax.set_ylabel(r'$y$', fontsize=12)

    #plt.show()

    # Preprocess f to construct Cluster object.
    M_csc = csc_matrix(M)
    M_bar = abs(M_csc)
    M_p = (M_csc + M_bar) / 2
    M_n = - (M_csc - M_bar) / 2

    M_p.eliminate_zeros()
    M_n.eliminate_zeros()

    C = Cluster((M_p, M_n))

    pcapreds = C.SPONGE_sym(2)
    return pcapreds