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


## Run the approximate bipartipe graph and the greedy algorithm a hundred times and save the results 
M = 50 # number of run 
nsize = 10 #size of the graph

#create vector to save the results in
res_diff = np.array([]) #the difference in results
greedy_score = np.array([]) #the greedy score 
quantum_score = np.array([]) #the quantum score 

for i in range(0,M):
    ## Classical algorithm - greedy 
    G = stochastic_block_model(nsize,a=0.1,b=0.5,c=0.1,d=0.1)
    _,_,Sg = greedy_algorithm(G)
    greedy_score = np.append(greedy_score,Sg)

    ## Quantum Annealing
    ## ------- Set up our QUBO dictionary -------
    Q = defaultdict(int)
    ## Update Q matrix for every edge in the graph
    for i, j in G.edges:
        Q[(i,i)]+= -1
        Q[(j,j)]+= -1
        Q[(i,j)]+= 2
    # # ------- Run our QUBO on the QPU -------
    # Set up QPU parameters
    chainstrength = 8
    numruns = 5
    # # Run the QUBO on the solver from your config file
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample_qubo(Q,
                                chain_strength=chainstrength,
                                num_reads=numruns,
                                label='Example - Maximum Cut')

    en_quant_list = []
    for sample, E in response.data(fields=['sample','energy']):
        en_quant_list.append(E)
    # print("engergy list ")
    # print(en_quant_list)
    max_en = - 1*min(en_quant_list)
    # print("Maximum energy")
    # print(max_en)
    quantum_score = np.append(quantum_score,max_en)

    ## Compute the difference
    diff = max_en-Sg
    # print("difference between greedy alg and quant annealing")
    # print(diff)
    res_diff = np.append(res_diff,diff)
    
name_folder1 = "results_undirected/SpectralValM="+str(M)+"-nodes=10-numruns="+str(numruns)+".csv"
name_folder2 = "results_undirected/QuantumValM="+str(M)+"-nodes=10-numruns="+str(numruns)+".csv"
savetxt(name_folder1, spectral_score, delimiter=',')
savetxt(name_folder2, quantum_score, delimiter=',')
## Save the data to work with it if necessary later 


## Histograms
mxres = int(res_diff.max())
mnres = int(res_diff.min())
fig1, ax1 = plt.subplots()
# f,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
hist, bin_edges = np.histogram(res_diff, range=(mnres, mxres), bins= mxres-mnres+1)
# An "interface" to matplotlib.axes.Axes.hist() method
n, bins,_ = plt.hist(x=res_diff, bins=bin_edges)
ax1.set_xlabel('Values of maximum cut')
ax1.set_ylabel('Frequency')
filename = "results_maxcut/maxcut_hist.png"
fig1.savefig(filename, bbox_inches='tight')
# print("finish")
# print("hist")
# print(hist)

## Scatter plot 
## plot the scatter plot
fig2, ax2 = plt.subplots()
#ax1.xlabel('Scatter plot of maximum cut')
maxplot = max(max(quantum_score),max(greedy_score))+3
## scatter plot needs to be perturbed a bit otherwise, can't see results 
quantum_scoreplot = quantum_score + np.random.rand(M)/2
greedy_scoreplot = greedy_score + np.random.rand(M)/2
ax2.scatter(x=quantum_scoreplot, y = greedy_scoreplot,s=30,marker="x")
x=np.linspace(0,maxplot,100)
ax2.plot(x,x,label="y=x")
ax2.set_ylabel('Greedy algorithm score')
ax2.set_xlabel('Quantum annealing score')
ax2.set_xlim([0, maxplot])
ax2.set_ylim([0, maxplot])
ax2.legend()
ax2.grid()
filename = "results_maxcut/maxcut_scatter.png"
fig2.savefig(filename, bbox_inches='tight')
print("done")
