import numpy as np


def _calculate_centroids(clustering):
    centroids= []
    for cluster in clustering:
        centroids.append(np.sum(cluster)/cluster.shape[0])

    return centroids

def _calculate_dissim_matrix(clustering):
    cens = _calculate_centroids(clustering)
    mat=np.ndarray(shape=(len(clustering),len(clustering)))
    for i in range(len(clustering)):
        for j in range(len(clustering)):
            mat[i][j]=np.linalg.norm(cens[i][j])

    return mat

def _calculate_cum_dist(clustering):
    dist=np.ndarray(shape=len(clustering))
    i=0
    cens=_calculate_centroids(clustering)
    for cluster in clustering:
        su = 0
        for node in cluster:
            su += np.linalg.norm(node,cens[i])
        dist[i]=su
        i+=1



