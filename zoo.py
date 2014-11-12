from scipy.ndimage.interpolation import zoom
import pandas as pd
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

gmap = gsomap(SP=0.9,dims=16,nr_s=3,lr_s=0.5)

for i in range(positions.shape[0]):
    positions [i]= gmap.process_input(features[i]).astype(int)

#print positions

names=np.column_stack((names,positions[:,0],positions[:,1]))
print names



labels = names[:,1]
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

plt.show()