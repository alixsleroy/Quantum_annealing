# ------ Import necessary packages ----
from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import networkx as nx
import numpy as np

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt



# ------- Set up maximum cut algorithm -------
def max_cut(G,S0_try1,S1_try1):
    maxcut_try1 =0
    for i,j in G.edges:
        if (i in S0_try1 and j in S1_try1) or (j in S0_try1 and i in S1_try1):
            maxcut_try1 =maxcut_try1+1
    return(maxcut_try1)
   
# ------- Set up the greedy algorithm to find the maximum cut -------
def greedy_algorithm(G):
    '''
    The function takes as an argument: 
    - G: a graphx value representing a graph
    The function returns: 
    - the two sets with allocations of nodes
    - the energy function value resulting from the allocation
    '''
    S0 = [] #empty set 1
    S1 = [] #empty set 2
    
    # Allocate the first vertex to any of the two groups randomly
    rand=np.random.normal(1)
    if rand<0:
        S0.append(list(G.nodes)[0])
    else:
        S1.append(list(G.nodes)[0])

    # Run through the rest of the nodes 
    for node in list(G.nodes)[1::]:    

        # Allocate the node to group 0 and compute max cut
        S0_try0 = S0.copy()
        S1_try0 = S1.copy()
        S0_try0.append(node)
        max_cut_S0= max_cut(G,S0_try0,S1_try0) # max cut for this allocation

        # Allocate the node to group 1 and compute max cut
        S0_try1 = S0.copy()
        S1_try1 = S1.copy()
        S1_try1.append(node)
        max_cut_S1= max_cut(G,S0_try1,S1_try1) # max cut for this allocation

        # Allocate to the group that yield the maximum cut
        if max_cut_S0>max_cut_S1:
            S0.append(node)
        else:
            S1.append(node)
        #elif max_cut_S0==max_cut_S1: #when same group, get a 50% change to be in each group

        #Compute the maximum cut
        score = max_cut(G,S0,S1)
    
    return(S1,S0,score)
    
