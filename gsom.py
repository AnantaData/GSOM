import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import Infinity, Inf, shape
from scipy.spatial.distance import minkowski, jaccard
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


    def __init__(self,SP=0.5,dims=3,nr_s=6,lr_s=0.9,boolean=False, lrr =0.5,fd=0.5):
        self.boolean=boolean
        self.fd=fd
        self.lrr = lrr
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
            self.nr=self.nr*self.lr
            self.lr = self.lr*self.lrr*(1-3.8/len(self.map_neurons.values()))
            if self.nr <=4 :
                return
            #normalization attempt:
            #for hsk in self.map_neurons.keys():
            #    neu = self.map_neurons[hsk]
            #    neu.weight_vs=0.5*neu.weight_vs
            #    self.map_neurons[hsk]=neu

        return


    def jaccard_sim(self,nparray1, nparray2,sym=True):

        b1=nparray1.astype(bool)
        b2=nparray2.astype(bool)
        if sym:
            out = len(np.where(np.logical_and(b1,b2))[0])+len(np.where(np.logical_or(b1,b2)==False)[0])
        else:
            out= len(np.where(np.logical_and(b1,b2))[0])

        return out/len(b1)



    def process_input(self,input_np_array):

        bmu = self.getBMU(input_np_array)
        for neu in self.map_neurons.values():
            nhash = str(neu.x_c)+""+str(neu.y_c)
           # print "bmu: "+str(bmu.coords())
            #print "neu: "+str(neu.coords())nhash
            if minkowski(bmu.coords().astype(float), neu.coords().astype(float), 2) < self.nr:
                '''weight adjustment'''
                neu.weight_vs = neu.weight_vs + self.lr * (input_np_array-neu.weight_vs)
                neu.res_err += minkowski(neu.weight_vs, bmu.weight_vs, 2)
                self.map_neurons[nhash]=neu

        '''growth'''

        if bmu.res_err > self.thresh:
            neu = bmu
            down=str(neu.x_c)+str(int(neu.y_c)-1)
            up=str(neu.x_c)+str(int(neu.y_c)+1)
            left=str(int(neu.x_c)-1)+str(neu.y_c)
            right=str(int(neu.x_c)+1)+str(neu.y_c)
            nei_coords = np.array([down, up , left , right ] )
            nei_coordi = np.array([[(neu.x_c),(int(neu.y_c)-1)], [(neu.x_c),(int(neu.y_c)+1)], [(int(neu.x_c)-1),(neu.y_c)], [(int(neu.x_c)+1),str(neu.y_c)]] )
            p =0
            for coord in nei_coords:
                n=None
                try:
                    n= self.map_neurons[coord]
                except KeyError:
                    nwron=neuron(nei_coordi[p][0], nei_coordi[p][1], self.dim)
                    #case a) new node has two consecutive nodes on one of its sides
                    #tiroshan and lakmal please implement the code here
                    #case b) between two old nodes
                    nwron.weight_vs=np.ndarray(shape=(self.dim))
                    nwron.weight_vs.fill(0.5)
                    n=nwron
                n.res_err+=self.fd*neu.res_err
                self.map_neurons[coord]=n
                p+=1

            bmu.res_err=self.thresh/2
            self.map_neurons[str(bmu.x_c)+""+str(bmu.y_c)]=bmu
        return bmu.coords()


    def getBMU(self,input_nparray):
        minDist=9223372036854775807
        candidate= None
        for neu in self.map_neurons.itervalues():
            #print "input: "+str(input_nparray)
            #print "neuron: "+str (neu.weight_vs)
            if self.boolean:
                cand = jaccard(input_nparray, neu.weight_vs)
                if minDist> cand:
                    minDist = cand
                    candidate= neu
            else:
                cand=minkowski(input_nparray, neu.weight_vs, 2)
                if minDist> cand:
                #print "mindist:",minDist
                #print "cand:",cand
                    minDist = cand
                    candidate= neu

                #print "candidate'scoords",candidate.coords()
        return  candidate




