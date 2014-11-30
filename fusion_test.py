from machinefusion import machine_fusion_gsom

__author__ = 'lakmal'

import numpy as np
import numpy.random as npr
import numpy as np
from gsom import gsomap
import matplotlib.pyplot as plt

data = np.loadtxt("zoo.data.txt",dtype=str,delimiter=",")
data = np.array(data)
names = data[:,0]
names= np.column_stack((names,data[:,-1]))
features= data[:,:-1]
features = features[:,1:].astype(int)
positions = np.ndarray(shape=(101,2))

gsom = gsomap(SP=0.9,dims=16,nr_s=4,lr_s=0.9,fd=0.99999)

gmap = machine_fusion_gsom(features,gsom,25,5,0.8,0.75)

keys = gmap.map_neurons.keys()
for k in keys:
    print gmap.map_neurons[k].x_c


for i in range(positions.shape[0]):
    positions [i]= gmap.process_input(features[i]).astype(int)
    #print positions[i]

names=np.column_stack((names,positions[:,0],positions[:,1]))

#print names

classification=np.array(['mammal','bird','reptile','fish','amphibian','insect','seacreature'])



labels = names[:,0]
#for i in range(labels.shape[0]):
#    labels[i]=classification[int(labels[i])-1]

plt.subplots_adjust(bottom = 0.1)
plt.scatter(
    positions[:, 0], positions[:, 1], marker = 'o', )
for label, x, y in zip(labels, positions[:, 0], positions[:, 1]):
    plt.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
'''
i=1
for neu in gmap.map_neurons.values():
    si = np.linalg.norm(neu.coassoc_vs)
    if si > 0:
        print "coassoc:"+str(i)+" :"+str(si)
        i=i+1

print "eliminated "+str(len(gmap.map_neurons)-i)+" out of "+str(len(gmap.map_neurons))+" neurons!"
'''
plt.show()

colors={1:"green", 2:"yellow", 3:"black", 4:"blue", 5:"red", 6:"orange", 7:"gray"}
colorlist=[]
for x in names[:,1]:
    colorlist.append(colors[int(x)])

colorlist=np.array(colorlist)

sizes = [20*2**2 for n in range(len(names[:,1]))]

plt.scatter(positions[:,0],positions[:,1],c=colorlist,s=sizes)
plt.show()