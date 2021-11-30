# ------ Import necessary packages ----
from collections import defaultdict

# from dwave.system.samplers import DWaveSampler
# from dwave.system.composites import EmbeddingComposite
import networkx as nx

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

# Primary data

name = 'Carlos M 12 Eric M 14 Dylan M 30 Friday M 41 Nicky M 48 Wilson M 49 Boris M 51 Tina F 8 Patti F 20 Chrissie F 21 Vila F 22 Zee-Zee F 23 Layla F 25 Alice F 26 Sally F 29 Sarah F 31 Mandy F 40 Farthing F 42 Rosie F 44 '
name = name.split()
#name = [n for n in name if n !='M' and n !='F']
name = name[::3]
print(name)

set_Male = name[:7]
print(set_Male)

set_Hand = ['Friday', 'Nicky', 'Wilson', 'Boris', 'Mandy', 'Rosie']
print(set_Hand)

def chimp_grooming():
    
    '''
    Construct chimp grooming network.
    '''
    
    # Create empty graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    G.add_nodes_from(name)
    
    # Add edges
    G.add_edge('Zee-Zee', 'Tina')
    G.add_edge('Zee-Zee', 'Mandy')
    
    G.add_edge('Tina', 'Zee-Zee')
    
    G.add_edge('Mandy', 'Tina')
    G.add_edge('Mandy', 'Dylan')
    G.add_edge('Mandy', 'Chrissie')
    
    G.add_edge('Layla', 'Mandy')
    
    G.add_edge('Farthing', 'Layla')
    G.add_edge('Farthing', 'Boris')
    
    G.add_edge('Rosie', 'Vila')
    G.add_edge('Rosie', 'Boris')
    
    G.add_edge('Vila', 'Eric')
    
    G.add_edge('Eric', 'Vila')
    G.add_edge('Eric', 'Patti')
    G.add_edge('Eric', 'Dylan')
    
    G.add_edge('Carlos', 'Alice')
    G.add_edge('Carlos', 'Dylan')
    G.add_edge('Carlos', 'Sarah')
    
    G.add_edge('Dylan', 'Eric')
    G.add_edge('Dylan', 'Carlos')
    
    G.add_edge('Friday', 'Wilson')
    G.add_edge('Friday', 'Boris')
    G.add_edge('Friday', 'Sally')
    G.add_edge('Friday', 'Sarah')
    G.add_edge('Friday', 'Dylan')
    
    G.add_edge('Sally', 'Friday')
    G.add_edge('Sally', 'Sarah')
    
    G.add_edge('Sarah', 'Friday')
    G.add_edge('Sarah', 'Sally')
    G.add_edge('Sarah', 'Nicky')
    G.add_edge('Sarah', 'Dylan')
     
    G.add_edge('Nicky', 'Sarah')
    G.add_edge('Nicky', 'Chrissie')
    
    G.add_edge('Alice', 'Sally')
    G.add_edge('Alice', 'Chrissie')
    G.add_edge('Alice', 'Carlos')
    
    G.add_edge('Chrissie', 'Dylan')
    G.add_edge('Chrissie', 'Nicky')
  
    
    
    nx.draw_networkx(G) 
    

    return G

#G = chimp_grooming()

def chimp_affiliative():
    '''
    Construct chimp affiliative network.
    '''
    
    # Create empty graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    G.add_nodes_from(name)
    
    # Add edges
    G.add_edge('Zee-Zee', 'Tina')
    G.add_edge('Zee-Zee', 'Dylan')
    
    G.add_edge('Tina', 'Zee-Zee')
    G.add_edge('Tina', 'Patti')
    G.add_edge('Tina', 'Vila')
    G.add_edge('Tina', 'Mandy')
    G.add_edge('Tina', 'Farthing')
    G.add_edge('Tina', 'Friday')
    G.add_edge('Tina', 'Layla')
    G.add_edge('Tina', 'Carlos')

    
    G.add_edge('Mandy', 'Tina')
    G.add_edge('Mandy', 'Dylan')
    G.add_edge('Mandy', 'Chrissie')
    
    G.add_edge('Layla', 'Tina')
    G.add_edge('Layla', 'Dylan')
    
    G.add_edge('Farthing', 'Zee-Zee')
    G.add_edge('Farthing', 'Dylan')
    
    #G.add_edge('Rosie', 'Vila')
    
    
    G.add_edge('Vila', 'Tina')
    G.add_edge('Vila', 'Dylan')
    
    #G.add_edge('Eric', 'Vila')
    
    G.add_edge('Carlos', 'Tina')
    G.add_edge('Carlos', 'Zee-Zee')

    G.add_edge('Dylan', 'Vila')
    G.add_edge('Dylan', 'Mandy')
    G.add_edge('Dylan', 'Sally')
    G.add_edge('Dylan', 'Alice')
    G.add_edge('Dylan', 'Zee-Zee')
    G.add_edge('Dylan', 'Layla')
    G.add_edge('Dylan', 'Carlos')
    
    G.add_edge('Friday', 'Tina')
    G.add_edge('Friday', 'Dylan')
    
    G.add_edge('Sally', 'Rosie')
    
    G.add_edge('Sarah', 'Patti')

    G.add_edge('Alice', 'Dylan')
    
    G.add_edge('Chrissie', 'Dylan')
  
    
    
    nx.draw_networkx(G) 
    

    return G

#G = chimp_affiliative()

def chimp_agonistic():
    
    ''' Construct chimp agonistic graph.
    '''
    
    # Create empty graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    G.add_nodes_from(name)
    
    # Add edges
    
    G.add_edge('Eric', 'Wilson')
    G.add_edge('Eric', 'Tina')
    G.add_edge('Eric', 'Patti')
    G.add_edge('Eric', 'Boris')
    G.add_edge('Eric', 'Friday')
    G.add_edge('Eric', 'Alice')
    G.add_edge('Eric', 'Dylan')
    
    G.add_edge('Carlos', 'Nicky')
    G.add_edge('Carlos', 'Dylan')
    G.add_edge('Carlos', 'Sally')
    G.add_edge('Carlos', 'Friday')
    G.add_edge('Carlos', 'Boris')
    
    G.add_edge('Dylan', 'Zee-Zee')
    G.add_edge('Dylan', 'Layla')
    G.add_edge('Dylan', 'Mandy')
    G.add_edge('Dylan', 'Wilson')
    G.add_edge('Dylan', 'Tina')
    G.add_edge('Dylan', 'Patti')
    G.add_edge('Dylan', 'Eric')
    G.add_edge('Dylan', 'Boris')
    G.add_edge('Dylan', 'Friday')
    G.add_edge('Dylan', 'Alice')
    G.add_edge('Dylan', 'Carlos')
    G.add_edge('Dylan', 'Nicky')
    
    G.add_edge('Sally', 'Dylan')
    G.add_edge('Sally', 'Alice')
    
    nx.draw_networkx(G) 
    

    return G

#G = chimp_agonistic()