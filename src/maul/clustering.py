from sklearn.cluster import SpectralClustering
from numpy.random import RandomState
import numpy as np

def wconsensus(P, W, k, random_state=None):
    """
    Parameters:
    `P` list: Partitions [y1, y2,..., yk]
    `W` list: Weights to be applied
    `k` int: Number of clusters
    `random_state` int: Random seed
    
    Return:
    `labels`: array
    
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