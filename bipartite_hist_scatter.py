# ------ Import necessary packages ----
from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import networkx as nx
import numpy as np
from numpy import savetxt

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

# from graph_creation import compexact_bipartipe_graph, stochastic_block_model
from greedy_algAlix import greedy_algorithm
from synthetic_data import stochastic_block_model
from synthetic_data import compexact_bipartipe_graph

from matplotlib import pyplot, patches
import numpy as np
from signet.cluster import Cluster
from scipy.sparse import csc_matrix
from spectral_alg import spectual


## Run the approximate bipartipe graph and the greedy algorithm a hundred times and save the results 
M = 1 #number of run 
nsize = 10 #size of the graph

#create vector to save the results in
res_diff = np.array([]) #the difference in results
spectral_score = np.array([]) #the greedy score 
quantum_score = np.array([]) #the quantum score 

for i in range(0,M):
    ## Classical algorithm - spectral
    #G = compexact_bipartipe_graph(10)
    G = stochastic_block_model(nsize,a=0.1,b=0.5,c=0.1,d=0.1)
    _,_,Sg = spectual(G)
    spectral_score = np.append(spectral_score,-Sg)
    print("Spectral algorithm")
    print(-Sg)
    ## Quantum Annealing
    ## ------- Set up our QUBO dictionary -------
    Q = defaultdict(int)
    ## Update Q matrix for every edge in the graph
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
                                label='Example - Bipartite structure')

    en_quant_list = []
    for sample, E in response.data(fields=['sample','energy']):
        en_quant_list.append(E)
    print("engergy list ")
    print(en_quant_list)
    max_en = -1*(min(en_quant_list) + int(G.number_of_edges()))
    print("Maximum energy")
    print(max_en)
    quantum_score = np.append(quantum_score,max_en)

    ## Compute the difference
    diff = max_en-Sg
    res_diff = np.append(res_diff,diff)
    
## Save the data to work with it if necessary later 
name_folder1 = "results_bipartite/SpectralValM="+str(M)+"-nodes="+str(nsize)+".csv"
name_folder2 = "results_bipartite/QuantumValM="+str(M)+"-nodes="+str(nsize)+".csv"
savetxt(name_folder1, spectral_score, delimiter=',')
savetxt(name_folder2, quantum_score, delimiter=',')

## Histograms
mxres = int(res_diff.max())
mnres = int(res_diff.min())
fig1, ax1 = plt.subplots()
# f,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
hist, bin_edges = np.histogram(res_diff, range=(mnres, mxres), bins= mxres-mnres+1)
# An "interface" to matplotlib.axes.Axes.hist() method
n, bins,_ = plt.hist(x=res_diff, bins=bin_edges)
ax1.set_xlabel('Bipartite graphs: differences')
ax1.set_ylabel('Frequency')
filename = "results_bipartite/bipartite_hist.png"
fig1.savefig(filename, bbox_inches='tight')


## Scatter plot 
## plot the scatter plot
fig2, ax2 = plt.subplots()
#ax1.xlabel('Scatter plot of maximum cut')
maxplot = max(max(quantum_score),max(spectral_score))+3
## scatter plot needs to be perturbed a bit otherwise, can't see results 
quantum_scoreplot = quantum_score + np.random.rand(M)/2
spectral_scoreplot = spectral_score + np.random.rand(M)/2
ax2.scatter(x=quantum_scoreplot, y = spectral_scoreplot,s=30,marker="x")
x=np.linspace(0,maxplot,100)
ax2.plot(x,x,label="y=x")
ax2.set_ylabel('Spectral algorithm score')
ax2.set_xlabel('Quantum annealing score')
ax2.set_xlim([0, maxplot])
ax2.set_ylim([0, maxplot])
ax2.legend()
ax2.grid()
filename = "results_bipartite/bipartite_scatter.png"
fig2.savefig(filename, bbox_inches='tight')
print("done")
