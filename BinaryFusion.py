__author__ = 'laksheen'

import numpy as np
from gsom import gsomap
import matplotlib.pyplot as plt
from scipy.spatial.distance import jaccard
from sklearn.metrics import jaccard_similarity_score

data = np.loadtxt("zoo.data.txt",dtype=str,delimiter=",")

data = np.array(data)

names = data[:,0]

names= np.column_stack((names,data[:,-1]))

features= data[:,:-1]

features = features[:,1:].astype(int)

print features.shape

positions = np.ndarray(shape=(101,2))

gmap1 = gsomap(SP=0.9,dims=16,nr_s=4,lr_s=0.9,fd=0.99999)
gmap2 = gsomap(SP=0.9,dims=16,nr_s=4,lr_s=0.9,fd=0.99999)
gmap3 = gsomap(SP=0.9,dims=16,nr_s=4,lr_s=0.9,fd=0.99999)
gmap4 = gsomap(SP=0.9,dims=16,nr_s=4,lr_s=0.9,fd=0.99999)


gmap1.process_batch(features,750)
gmap2.process_batch(features,750)
gmap3.process_batch(features,750)
gmap4.process_batch(features,750)

coassocs={}
for neu in gmap1.map_neurons.values():
    coassocs[1,str(neu.coords()[0])+""+str(neu.coords()[1])]=neu.binarycoassoc_vs

for neu in gmap2.map_neurons.values():
    coassocs[2,str(neu.coords()[0])+""+str(neu.coords()[1])]=neu.binarycoassoc_vs

for neu in gmap3.map_neurons.values():
    coassocs[3,str(neu.coords()[0])+""+str(neu.coords()[1])]=neu.binarycoassoc_vs

for neu in gmap4.map_neurons.values():
    coassocs[4,str(neu.coords()[0])+""+str(neu.coords()[1])]=neu.binarycoassoc_vs

#print len(coassocs.keys())

usage_threshold = 5

for map_coords in coassocs.keys():

    if np.sum(coassocs[map_coords]) < usage_threshold:
        del coassocs[map_coords]

#print len(coassocs.keys())

fusion_threshold = 0.97

n = len(coassocs.keys())
incidence_matrix = np.zeros(shape=(n,n))

print incidence_matrix.shape

#print len(incidence_matrix)

groups=np.zeros(len(coassocs.keys()))

for i in range(len(coassocs.keys())-1):

    checkee = coassocs[coassocs.keys()[i+1]]

    if i == len(coassocs.keys())-2:
        incidence_matrix[i+1][i+1] = 1

    incidence_matrix[i][i] = 1

    for j in range(i+1):
        if jaccard_similarity_score(checkee,coassocs[coassocs.keys()[j]]) > fusion_threshold :
            incidence_matrix[j][i+1] = 1

print incidence_matrix

for k in range(len(incidence_matrix)):
    if(k==0):
        group_num =1
    else:
        if(groups[k]==0):
            group_num +=1
        else:
            group_num = groups[k]

    for j in range(k,len(incidence_matrix)):
        if (incidence_matrix[k][j] == 1):
            groups[j] = group_num

#print groups

m = len(np.unique(groups))

clusters = {}

for i in range(len(coassocs.keys())):
    map_no = coassocs.keys()[i][0]
    coorstr = coassocs.keys()[i][1]
    weight=None
    gmap = None

    if map_no == 1:
        gmap=gmap1
    elif map_no == 2:
        gmap = gmap2
    elif map_no == 3:
        gmap = gmap3
    elif map_no == 4:
        gmap = gmap4

    weight = gmap.map_neurons[coorstr].weight_vs

    try :
        temp = clusters[groups[i]-1]
    except:
        temp=[]
    temp.append(weight)
    clusters[groups[i]-1]=temp

#print clusters

w = []

for i in clusters.keys():
    w.append( sum(np.array(clusters[i]))/np.array(clusters[i]).shape[0] )


print w
