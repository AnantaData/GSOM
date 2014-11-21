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

coassocs={}
for neu in gmap1.map_neurons.values():
    coassocs[1,str(neu.coords()[0])+""+str(neu.coords()[1])]=neu.coassoc_vs

for neu in gmap2.map_neurons.values():
    coassocs[2,str(neu.coords()[0])+""+str(neu.coords()[1])]=neu.coassoc_vs

for neu in gmap3.map_neurons.values():
    coassocs[3,str(neu.coords()[0])+""+str(neu.coords()[1])]=neu.coassoc_vs

for neu in gmap4.map_neurons.values():
    coassocs[4,str(neu.coords()[0])+""+str(neu.coords()[1])]=neu.coassoc_vs

count = 1

used ={}

for map_coords in coassocs.keys():


    if np.linalg.norm(coassocs[map_coords])>0.1:
        print count+1
        count+=1
        used[map_coords]=coassocs[map_coords]


#used=np.array(used)
############ the new coassociation vectors are good enough I guess.. Let's go for the similarity measure and stuff

fusion_threshold = 0.1

done = np.zeros(len(used.keys()),dtype=bool)
group_num = 1

groups=np.zeros(len(used.keys()))
groups[0]=1
done[0]=True
for i in range(len(used.keys())-1):
    found=False
    checkee = used[used.keys()[i+1]]
    for j in range(i):
        if np.linalg.norm(checkee-used[used.keys()[j]])<fusion_threshold:
            groups[i+1]=groups[j]
            found=True
    if found == False:
        group_num+=1
        groups[i+1]=group_num


#print len(np.unique(groups))

clusters = {}

for i in range(len(used.keys())):
    map_no = used.keys()[i][0]
    coorstr = used.keys()[i][1]
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



#print len(clusters[10])
w = []

for i in clusters.keys():
    w.append( sum(np.array(clusters[i]))/np.array(clusters[i]).shape[0] )

print np.array(w).shape

