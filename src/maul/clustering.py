##### Consensus Index #####

from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score
from numpy.random import RandomState
import numpy as np

def consensus_score(P):
    """
    Parameters:
    `P` list: Partitions [y1, y2,..., yk]

    Return:
    `consensus_score`: float
    
    """
        
    scores = []
    for i in range(len(P)):
        for j in range(i+1, len(P)):
            scores.append(adjusted_rand_score(P[i], P[j]))
            
    return sum(scores)

def get_onehot(arr):
    """
    Parameters:
    `arr`: array to encode

    Return:
    `encoded_array`: encoded array
    
    """
    
    encoded_array = np.zeros((arr.size, arr.max()+1), dtype=int)
    encoded_array[np.arange(arr.size), arr] = 1 

    return encoded_array

def wconsensus(P, W, k, random_state=None):
    """
    Parameters:
    `P` list: Partitions [y1, y2,..., yk]
    `W` list: Weights to be applied
    `k` int: Number of clusters
    `random_state` int: Random seed
    
    Return:
    `label`: array
    
    """
    
    assert len(P) == len(W)
    
    if random_state is None:
        random_state = RandomState()
    elif type(random_state) == int:
        random_state = RandomState(random_state)
    else:
        assert type(random_state) == RandomState
        
    H_ = np.concatenate((get_onehot(P[0])*W[0], get_onehot(P[1])*W[1]), axis=1)

    for i in range(2, len(P)):
        H_ = np.concatenate((H_, get_onehot(P[i])*W[i]), axis=1)

    S_ = (H_ @ H_.transpose())
    clusters = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=random_state).fit(S_)
    
    return clusters.labels_
    
##### Clustering Utility Based on Averaged information Gain of isolating Each cluster (CUBAGE)#####

import pandas as pd
from scipy.stats import entropy

def cubage_score(X, label):

    X = pd.DataFrame(X)
    k_ = max(label)+1

    HU = sum(entropy(X))
    X['k'] = label

    H = []
    W = []
    for i in range(k_):
        H.append(entropy(X[X.k == i].loc[:, X[X.k == i].columns != 'k']))
        W.append(X[X.k == i].shape[0]/X.shape[0])

    H = np.array([sum(n) for n in H])
    W = np.array(W)

    HC = []
    for i in range(max(label)+1):
        HC.append(entropy(X[X.k != i].loc[:, X[X.k != i].columns != 'k']))

    HC = np.array([sum(n) for n in HC])

    E = W@H
    AGE = HU - 1/k_*(E + (1-W)@HC)
    CUBAGE = AGE/E
    
    return CUBAGE


##### Density-core-based Clustering Validation Index (DCVI) #####

def MST(G, show=False): # Prim's Algorithm
    """
    Parameters:
    `G` array: Adjacency matrix
    `show` bool: Show the vertices with the weights or not 
    
    Return:
    `w` array: Vertex weights
    
    """
    w=[]
    INF = 9999999
    V = len(G)
    selected = [0]*V
    no_edge = 0
    selected[0] = True
    
    if show == True:
        print("Edge:Weight")
    
    while (no_edge < V - 1):
        minimum = INF
        x = 0
        y = 0
        for i in range(V):
            if selected[i]:
                for j in range(V):
                    if ((not selected[j]) and G[i][j]):  
                        if minimum > G[i][j]:
                            minimum = G[i][j]
                            x = i
                            y = j
        if show == True:
            print(str(x) + "-" + str(y) + ":" + str(G[x][y]))
            
        w.append(G[x][y])
        selected[y] = True
        no_edge += 1
        
    return w