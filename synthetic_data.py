
# ------ Import necessary packages ----
from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import networkx as nx
import numpy as np

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

def stochastic_block_model(N,a=0.1,b=0.5,c=0.1,d=0.1):
    ## This function takes as arguments: 
    ##  N : number of nodes to put in the graph
    ##  a : Probability to have a directed edge between two nodes in A
    ##  b : Probability to have a directed edge between a node in A and a node in B 
    ##  c : Probability to have a directed edge between a node in B and a node in A 
    ##  d : Probability to have a directed edge between two nodes in B
    ## This function create a Nx directed graph G which has a non exact bipartipe structure. Nodes 1 to N/2 (N pair)
    ## belongs to A and nodes N/2+1 to N belongs to B.  
    ## This function returns: 
    ##  G: the graph created
    ## and draws the graph created with label on the nodes
    
 
    # Create empty graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for i in range(1,N):
        G.add_nodes_from([i,i+1])
    
    Nhalf = int(N/2)
        
    # Loop to allocate vertices to nodes
    for i in range(1,Nhalf+1): # Nodes from 0,1,..N/2 belongs to A ...
        for j in range(Nhalf+1,N+1):  #...and nodes from N/2+1,...,N belongs to B. 
            pAB = np.random.uniform(0,1) #get a random draw bw 0 and 1
            if pAB>(1-b): #if below the probality that a node goes from A to B
                G.add_edges_from([(i,j)]) #add a directed edge from i to j
        for j in range(1,Nhalf+1): #... and nodes from 0,1,...,N/2+1 belongs to A
            pAA = np.random.uniform(0,1) 
            if (j!=i) and pAA>(1-a): # additional conditions is i different from j (do not want to link nodes to themselves)
                G.add_edges_from([(i,j)]) #add a directed edge from i to j
    
    # Loop to allocate vertices to nodes
    for i in range(Nhalf+1,N+1): # Nodes from N/2+1,...,N belongs to B ...
        for j in range(1,Nhalf+1): # ...and nodes from 0,1,...N/2+1 belongs to A.  
            pBA = np.random.uniform(0,1) #get a random draw bw 0 and 1
            if pBA>(1-c): #if above the probality that a node goes from 
                G.add_edges_from([(i,j)]) #add a directed edge from i to j
        for j in range(Nhalf+1,N+1): #... and nodes from 0,1,...,N/2+1 belongs to A
            pBB = np.random.uniform(0,1) 
            if (j!=i) and pBB>(1-d): # additional conditions is i different from j (do not want to link nodes to themselves)
                G.add_edges_from([(i,j)]) #add a directed edge from i to j
           
                
    #Draw the graph created
    setA = list(range(1,Nhalf+1))
    pos=nx.spring_layout(G) # Draw the directed edges
    nx.draw_networkx(G,pos=nx.bipartite_layout(G, setA)) #Draw the directed edges and nodes
    labels = nx.get_edge_attributes(G,'weight') #create the labels (name of the nodes)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels) #add the label to the plot
    filename = "stoch_bloc.png"
    plt.savefig(filename, bbox_inches='tight')

    return(G)


def compexact_bipartipe_graph(N):
    ## This function takes as arguments: 
    ##  N : number of nodes to put in the graph
    ## This function create a Nx directed graph G which has a complete bipartipe structure. The direction of the edge is from 
    ## A to B, where nodes 1 to N/2 belongs to A and nodes N/2+1 to N belong to B. 
    ## This function returns: 
    ##  G: the graph created
    ## and draws the graph created with label on the nodes
        
    # Create empty graph
    G = nx.DiGraph()
    # Add nodes to the graph
    for i in range(1,N):
        G.add_nodes_from([i,i+1])

    Nhalf = int(N/2)
    # Loop to allocate vertices to nodes
    for i in range(1,Nhalf+1): # Nodes from 0,1,..N/2 belongs to A ...
        for j in range(Nhalf+1,N+1):  #...and nodes from N/2+1,...,N belongs to B. 
            G.add_edges_from([(i,j)]) #add a directed edge from i to j
 
    # Draw the graph created
    setA = list(range(1,Nhalf+1))
    pos=nx.spring_layout(G) # Draw the directed edges
    nx.draw_networkx(G,pos=nx.bipartite_layout(G, setA)) #Draw the directed edges and nodes
    labels = nx.get_edge_attributes(G,'weight') #create the labels (name of the nodes)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels) #add the label to the plot
    filename = "completeproblem.png"
    plt.savefig(filename, bbox_inches='tight')
    return(G)
