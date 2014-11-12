import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import Infinity, Inf, shape
from scipy.spatial.distance import minkowski
from Crypto.Util.number import size

class neuron(object):

    x_c = 0
    y_c = 0
    res_err = 0
    weight_vs=None

    def __init__(self,x,y,dims):
        self.x_c=x
        self.y_c=y
        self.weight_vs=np.random.random(size=dims)
       # print str(x)+","+str(y)+":"+str(self.weight_vs)

    def coords(self):
        self.cors= [str(self.x_c), str(self.y_c)]
        return np.array(self.cors)


class gsomap(object):

    map_neurons ={}

    def viewmap(self):
        x=np.ndarray(shape=len(self.map_neurons))
        y=np.ndarray(shape=len(self.map_neurons))
        i=0;
        for neu in self.map_neurons.itervalues():
            x[i]=neu.x_c
            y[i]=neu.y_c
            i+=1

        plt.plot_date(x, y)
        plt.xlabel("xcoords")
        plt.ylabel("ycoords")
        plt.show()


    def __init__(self,SP=0.5,dims=3,nr_s=6,lr_s=0.9):
        self.dim = dims
        for i in range(4):
            x=i/2
            y=i%2
            nhash = str(x)+""+str(y)
            self.map_neurons[nhash] = neuron(i/2, i%2, dims)

        ''' definition of growth threshold according to GSOM paper'''

        self.thresh=-1*dims*np.log(SP)
        self.nr=nr_s
        self.lr=lr_s
        #print self.map_neurons


    def process_batch(self,batch_np_array, k=10):
        for j in range(k):
            for i in range(batch_np_array.shape[0]):
                tinp = batch_np_array[i]
                self.process_input(tinp)
            self.nr=self.nr*self.lr/k
            if self.nr <=1 :
                return
        return


 #   def adjust(self,bmu_coord_nparr,neu):
 #       n=neuron(neu)
  #      if minkowski(bmu_coord_nparr, n.coords(), 2)<self.nr:
   #         nei_coords = np.array([str(n.x_c)+str(n.y_c-1), str(n.x_c)+str(n.y_c+1), str(n.x_c-1)+str(n.y_c), str(n.x_c+1)+str(n.y_c)] )
    #        for nei_coord in nei_coords:
     #           neu = self.map_neurons[nei_coord]
      #          self.adjust(bmu_coord_nparr, neu)



    def process_input(self,input_np_array):

        bmu = self.getBMU(input_np_array)

        for neu in self.map_neurons.values():
           # print "bmu: "+str(bmu.coords())
            #print "neu: "+str(neu.coords())
            if minkowski(bmu.coords().astype(float), neu.coords().astype(float), 2) < self.nr:
                '''weight adjustment'''
                neu.weight_vs = neu.weight_vs + self.lr * bmu.weight_vs
                neu.res_err += minkowski(neu.weight_vs, bmu.weight_vs, 2)
                '''growth'''
                if neu.res_err > self.thresh:
                    down=str(neu.x_c)+str(int(neu.y_c)-1)
                    up=str(neu.x_c)+str(int(neu.y_c)+1)
                    left=str(int(neu.x_c)-1)+str(neu.y_c)
                    right=str(int(neu.x_c)+1)+str(neu.y_c)
                    nei_coords = np.array([down, up , left , right ] )
                    nei_coordi = np.array([[(neu.x_c),(int(neu.y_c)-1)], [(neu.x_c),(int(neu.y_c)+1)], [(int(neu.x_c)-1),(neu.y_c)], [(int(neu.x_c)+1),str(neu.y_c)]] )
                    p=0
                    for coord in nei_coords:
                        '''if no neighbor is found, grow'''
                        n=None
                        try:
                            n= self.map_neurons[coord]
                        except KeyError:
                            nwron=neuron(nei_coordi[p][0], nei_coordi[p][1], self.dim)
                            nwron.weight_vs=neu.weight_vs
                            n=nwron
                        n.res_err+=neu.res_err-self.thresh
                        self.map_neurons[coord]=n
                        p+=1
            #print "returning: ",bmu.coords()
        return bmu.coords()


    def getBMU(self,input_nparray):
        minDist=9223372036854775807
        candidate= None
        for neu in self.map_neurons.itervalues():
            #print "input: "+str(input_nparray)
            #print "neuron: "+str (neu.weight_vs)
            cand=minkowski(input_nparray, neu.weight_vs, 2)
            if minDist> cand:
                #print "mindist:",minDist
                #print "cand:",cand
                minDist = cand
                candidate= neu

                #print "candidate'scoords",candidate.coords()
        return  candidate




