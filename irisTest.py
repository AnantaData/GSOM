__author__ = 'tiroshan'
from sklearn import datasets
from gsom import gsomap
import numpy as np
import matplotlib.pyplot as plt
import sklearn

iris = datasets.load_iris()

iris_feed = np.array(iris.data)
iris_feed = sklearn.preprocessing.normalize(iris_feed)
positions = np.ndarray(shape=(150,2))

gmap = gsomap(SP=0.9,dims=4,nr_s=4,lr_s=0.9,fd=0.99999)
gmap.process_batch(iris_feed,800)

for i in range(positions.shape[0]):
    positions [i]= gmap.process_input(iris_feed[i])

plt.subplots_adjust(bottom = 0.1)
plt.scatter(
    positions[:, 0], positions[:, 1], marker = 'o', )
plt.show()

colors={0:"green", 1:"yellow", 2:"black"}
colorlist=[]
for x in iris.target:
    colorlist.append(colors[int(x)])

colorlist=np.array(colorlist)

sizes = [20*2**2 for n in range(len(iris.target))]

plt.scatter(positions[:,0],positions[:,1],c=colorlist,s=sizes)
plt.show()