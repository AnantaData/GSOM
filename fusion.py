import numpy as np
from gsom import gsomap
import matplotlib.pyplot as plt
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

coassocs=[]
for neu in gmap1.map_neurons.values():
    coassocs.append(neu.coassoc_vs)

for neu in gmap2.map_neurons.values():
    coassocs.append(neu.coassoc_vs)

for neu in gmap3.map_neurons.values():
    coassocs.append(neu.coassoc_vs)

for neu in gmap4.map_neurons.values():
    coassocs.append(neu.coassoc_vs)

count = 1

used =[]

for coassoc in coassocs:


    if np.linalg.norm(coassoc)>0.1:
        print count+1
        count+=1
        used.append(coassoc)
        print np.linalg.norm(coassoc)

used=np.array(used)
############ the new coassociation vectors are good enough I guess.. Let's go for the similarity measure and stuff



clusters={}
clusters[0]=[]
clusters[0].append(used[0])

fthresh=0.1

for i in range(used.shape[0]):
    for j in range(len(clusters.values())):
        cluster = clusters[j]
        for vec in cluster:
            if np.linalg.norm(used[i]-vec) < fthresh:
                cluster.append(used[i])
                clusters[i]=cluster
                break

        clusters[j+1]=[]
        clusters[j+1].append(used[i])

for i in range(len(clusters)):
    try:
        len(clusters[i])
        print i,len(clusters[i])
    except:
        print i
        break