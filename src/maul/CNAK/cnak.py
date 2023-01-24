import numpy as np
from numpy.random import RandomState
import munkres
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