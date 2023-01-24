import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score
from numpy.random import RandomState
import numpy as np
import pandas as pd

##### Consensus Index #####

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

def entropy_ajusted_arr(arr):
    """
    Parameters:
    `arr` array
    
    Return:
    `E`: Entropy
    
    """

    if len(np.shape(arr)) == 2:
        E = []
        for i in range(np.shape(arr)[1]):
            p1 = sum(arr[:,i])/len(arr[:,i])
            p0 = 1 - p1

            if (p1 == 0) | (p0 == 0):
                E.append(0)
            else:
                E.append(-((p0 * np.log2(p0)) + (p1 * np.log2(p1))))
                
    else:
        p1 = sum(arr)/len(arr)
        p0 = 1 - p1
        
        E = -((p0 * np.log2(p0)) + (p1 * np.log2(p1)))
        
    return E

def entropy_ajusted_df(df):
    """
    Parameters:
    `df` DataFrame
    
    Return:
    `E`: Entropy
    
    """
    if len(df.shape) == 2:
        E = []
        for i in range(df.shape[1]):
            p1 = sum(df[i])/len(df[i])
            p0 = 1 - p1

            if (p1 == 0) | (p0 == 0):
                E.append(0)
            else:
                E.append(-((p0 * np.log2(p0)) + (p1 * np.log2(p1))))

    else:
        p1 = sum(df)/len(df)
        p0 = 1 - p1

        E = -((p0 * np.log2(p0)) + (p1 * np.log2(p1)))

    return E

def cubage_score(X, labels):
    """
    Parameters:
    `X` array: Data
    `labels` array: Labels 
    
    Return:
    `CUBAGE` float: Score
    
    """
    
    assert type(X).__module__ == np.__name__
    assert type(labels).__module__ == np.__name__
    
    X = pd.DataFrame(X)
    k_ = max(labels)+1

    HU = sum(entropy_ajusted_df(X))
    X['k'] = labels
    
    H = []
    W = []
    for i in range(k_):
        H.append(entropy_ajusted_df(X[X.k == i].loc[:, X[X.k == i].columns != 'k']))
        W.append(X[X.k == i].shape[0]/X.shape[0])

    H = np.array([sum(n) for n in H])
    W = np.array(W)

    HC = []
    for i in range(k_):
        HC.append(entropy_ajusted_df(X[X.k != i].loc[:, X[X.k != i].columns != 'k']))

    HC = np.array([sum(n) for n in HC])

    E = W@H
    AGE = HU - 1/k_*(E + (1-W)@HC)
    CUBAGE = AGE/E
    
    return CUBAGE
    
##### Contiguous Density Region (CDR) Index #####

# Example

#import matplotlib.pyplot as plt

#X = np.array([[1,1], [2,1], [2,2], [1, 1.5],
#              [1,6], [2,6], [2, 5.5],
#              [7,5], [7,6], [8,5], [8,6], [7, 5.5],
#              [8,1], [8, 1.5],
#              [4.5,3.5]])

#labels = np.array([0,0,0,0,1,1,1,2,2,2,2,2,3,3,4])

#plt.scatter(X[:,0], X[:,1])
#plt.grid()
#plt.show    
        
def cdr_score(X, labels):
    """
    Parameters:
    `X` array: Data
    `labels` array: Labels 
    
    Return:
    `cdr` float: Score
    
    """

    assert type(X).__module__ == np.__name__
    assert type(labels).__module__ == np.__name__

    X = pd.DataFrame(X)
    k_ = max(labels)+1

    X['k'] = labels

    unif = []
    nk = []
    
    for k in range(k_):
        #print('K:', k) ############## <<<
        points = X[X.k == k].loc[:, X[X.k == k].columns != 'k'].reset_index(drop=True) #filter the points of a specific cluster
        nk.append(points.shape[0]) #get the number of examples in the each cluster
        
        df = pd.DataFrame(euclidean_distances(points, points)) #distance
        df.replace(0, np.nan, inplace=True)
        l = df.min(axis=0)
        
        avg_den_k = sum(l)/len(l) #get density of clusters
        #print('local_den:', l) ############## <<<
        #print('avg_den_k:', avg_den_k) ############## <<<
        
        if len(l) == 1:
            unif.append(0)
            #print('local_den - avg_den_k:', 0, '\n') ############## <<<
            
        else:
            unif.append(sum([abs(_- avg_den_k) for _ in l])/avg_den_k) #get uniformity of clusters
            #print('local_den - avg_den_k:', [abs(_- avg_den_k) for _ in l], '\n') ############## <<<

    cdr = sum([nk[i]*unif[i] for i in range(len(unif))])/sum(nk) #get Contiguous Density Region (CDR) index.
    #print('nk[i]*unif[i]:', [nk[i]*unif[i] for i in range(len(unif))]) ############## <<<
    #print('unif:', unif) ############## <<<
    #print('nk:', nk, '\n') ############## <<<
    
    return cdr
  
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
    

##### Cluster Number Assisted Kmeans (CNAK) #####

import numpy as np
from numpy.random import RandomState
from scr.maul import munkres
from sklearn.cluster import KMeans
import argparse

def Bucketization(cluster1,cluster2,indexes,turn,centers):
    if turn==0:
        for i in range(len(cluster1)):
            centers[i].append(cluster1[i])


    for i in range(len(cluster2)):
        r,c=indexes[i]
        centers[r].append(cluster2[c])

    return centers

def weightMatrix(clusters,dd_list,k_centers):
    count=0
    avg_score=0
    K=len(clusters[0])
    for i in range(len(clusters)):
        cluster1=clusters[i]
        for j in range(i+1,len(clusters)):
            cluster2=clusters[j]
            count=count+1
            weight=np.zeros([K,K])
            weight_bk=np.zeros([K,K])
            #~~~~~computation of distance between two sets of centroids~~~~~~~
            for m in range(len(cluster1)):

                vec1=np.asarray(cluster1[m])
                for n in range(len(cluster2)):
                    vec2=np.asarray(cluster2[n])
                    weight_bk[m][n]=np.linalg.norm(vec1-vec2)
                    weight[m][n]=np.linalg.norm(vec1-vec2)

            score=0
            #~~~~~Computation of perfect matching between two sets of centroids using Kuhn-Munkre's Algorithm~~~~~
            matching = munkres.Munkres()
            indexes = matching.compute(weight_bk)

            #~~~~~Similar centroids are put in same bucket~~~~~~
            if i==0:
                Bucketization(cluster1,cluster2,indexes,j,k_centers)

            #~~~~~~CNAK score~~~~~~~~~
            for r, c in indexes:
                score=score+weight[r][c]
            score=score/len(cluster1)
            avg_score=avg_score+score
            dd_list.append(score)

    avg_score=avg_score/count
    return avg_score, count, dd_list, k_centers

def weightMatrixUpdated(global_centroids_list,clusters,dd_list,k_centers,avg_score,count):

    avg_score=avg_score*count

    K=len(clusters[0])

    for i in range(len(global_centroids_list)):
        cluster1=global_centroids_list[i]
        for j in range(len(clusters)):
            cluster2=clusters[j]
            count=count+1
            weight=np.zeros([K,K])
            weight_bk=np.zeros([K,K])
            #~~~~~computation of distance between two sets of centroids~~~~~~~	
            for m in range(len(cluster1)):

                vec1=np.asarray(cluster1[m])
                for n in range(len(cluster2)):
                    vec2=np.asarray(cluster2[n])
                    weight_bk[m][n]=np.linalg.norm(vec1-vec2)
                    weight[m][n]=np.linalg.norm(vec1-vec2)

            score=0
            #~~~~~Computation of perfect matching between two sets of centroids using Kuhn-Munkre's Algorithm~~~~~
            matching = munkres.Munkres()
            indexes = matching.compute(weight_bk)
            #~~~~~Similar centroids are put in same bucket~~~~~~
            if i==0:
                Bucketization(cluster1,cluster2,indexes,j,k_centers)
            #~~~~~~CNAK score~~~~~~~~~
            for r, c in indexes:
                score=score+weight[r][c]
            score=score/len(cluster1)
            dd_list.append(score)
            avg_score=avg_score+score
    avg_score=avg_score/count
    return avg_score, count, dd_list, k_centers

def CNAK_core(data,gamma,K,random_state):
    
    if random_state is None:
        random_state = RandomState()
    elif type(random_state) == int:
        random_state = RandomState(random_state)
    else:
        assert type(random_state) == RandomState

    T_S=1
    T_E=5

    centroids_list=[]
    for j in range(T_S,T_E):
        #~~~~~random sampling without repitiion~~~~~~~~
        #print("int(len(data)*gamma):",int(len(data)*gamma))
        index=random_state.choice(range(0, len(data)),int(len(data)*gamma))
        samples=[]
        for k in range(int(len(data)*gamma)):
            temp=data[index[k]]
            samples.append(temp)
        #~~~~~~K-means++ on sampled dataset~~~~~~~~~
        kmeans = KMeans(n_clusters=K,init='k-means++',n_init=20,max_iter=300,tol=0.0001).fit(samples)
        centroids=kmeans.cluster_centers_
        centroids_list.append(centroids)	
    dd_list=[]
    k_centers=[[] for i in range(len(centroids))]
    #~~~~~~Computation of CNAK score and forming K buckets with T_E similar  centroids~~~~~~~~
    avg_score, count, dd_list,k_centers=weightMatrix(centroids_list,dd_list,k_centers)

    #~~~~~Estimate the value of T ~~~~~~~~~
    mean=np.mean(dd_list)
    std=np.std(dd_list)
    val=(1.414*20*std)/(mean)

    global_centroids_list=[]
    for centroids in (centroids_list):
        global_centroids_list.append(centroids)
    centers=[[] for i in range(len(centroids_list[0]))]

    #~~~~~Repeat untill  T_E > T_threshold ~~~~~~~~~
    while val>T_E:
        T_S=T_E
        T_E=T_E+1
        centroids_list=[]
        for j in range(T_S,T_E):
            index=random_state.choice(range(0, len(data)),int(len(data)*gamma))
            datax=[]

            for k in range(int(len(data)*gamma)):
                temp=data[index[k]]
                datax.append(temp)

            kmeans = KMeans(n_clusters=K,init='k-means++',n_init=20,max_iter=300,tol=0.0001).fit(datax)
            centroids=kmeans.cluster_centers_
            centroids_list.append(centroids)

        avg_score, count, dd_list,k_centers=weightMatrixUpdated(global_centroids_list,centroids_list,dd_list,k_centers,avg_score,count)
        for centroids in ((centroids_list)):
            global_centroids_list.append(centroids)
        mean=np.mean(dd_list)
        std=np.std(dd_list)
        val=(1.414*20*std)/(mean)

    clusterCenterAverage=[]
    for i in range(len(k_centers)):
          clusterCenterAverage.append(np.mean(k_centers[i],axis=0))
    return val, T_E, avg_score, clusterCenterAverage

def CNAK(data, gamma=0.7, k_min=2, k_max=21, random_state=None, show_scores=False):
    
    if random_state is None:
        random_state = RandomState()
    elif type(random_state) == int:
        random_state = RandomState(random_state)
    else:
        assert type(random_state) == RandomState

    CNAK_score=[]
    k_max_centers=[]

    for K in range(k_min,k_max):
        val, T_E, avg_score, k_centers=CNAK_core(data,gamma,K,random_state)
        CNAK_score.append(avg_score)
        k_max_centers.append(k_centers)

    K_hat=CNAK_score.index(min(CNAK_score))
    
    if show_scores == True:
        return K_hat+k_min, CNAK_score
    
    else:
        return K_hat+k_min