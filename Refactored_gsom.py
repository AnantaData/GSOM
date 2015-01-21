__author__ = 'Damith/laksheen'

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from numpy import Infinity, Inf, shape
from scipy.spatial.distance import minkowski, jaccard
from Crypto.Util.number import size
import sys
from joblib import Parallel, delayed

class neuron(object):

    x_c = 0
    y_c = 0
    res_err = 0
    weight_vs=None
    #new entry
    k_coassoc_vs = None
    binarycoassoc_vs = None

    def __init__(self,x,y,dims):
        """

        :rtype : object
        """
        self.x_c=x
        self.y_c=y
        self.weight_vs=np.random.random(size=dims)
        self.k_coassoc_vs = np.zeros(shape=150)
        self.binarycoassoc_vs = np.zeros(shape=150)
        self.inputs = []
        #print self.coassoc_vs
       # print str(x)+","+str(y)+":"+str(self.weight_vs)

    def coords(self):
        self.cors= [str(self.x_c), str(self.y_c)]
        return np.array(self.cors)


class gsom(object):
    map_neurons ={}

    def __init__(self,SP,dims,nr_s,lr_s,lrr,fd,n_jobs=2):
        self.boolean=False
        self.n_jobs= n_jobs
        self.thresh=-1*dims*np.log(SP)
        self.map_sizes = []
        self.nr = nr_s
        self.lr = lr_s
        self.lr_red_coef = lrr
        self.dim = dims
        self.fd = fd

        for i in range(4):
            x=i/2
            y=i%2
            nhash = str(x)+""+str(y)
            print nhash
            self.map_neurons[nhash] = neuron(i/2, i%2, dims)

    def _viewmap(self):
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
        return

    def _predict_point(self, input_array):
        bmu = self._get_BMU(input_array)
        return bmu.coords()

    def _predict_point_1(self, input_array, i):
        bmu = self._get_BMU(input_array)
        bmu.inputs.append(i)
        return bmu.coords()

    def _get_BMU(self,input_nparray):

        minDist=9223372036854775807
        candidate= None
        for neu in self.map_neurons.itervalues():

            if self.boolean:
                cand = jaccard(input_nparray, neu.weight_vs)
                if minDist> cand:
                    minDist = cand
                    candidate= neu
            else:
                cand=minkowski(input_nparray, neu.weight_vs, 2)
                if minDist> cand:
                    minDist = cand
                    candidate= neu

        return  candidate

    def _grow_map(self,input,k):
        bcoords = self._process_input(input)
        bhash=str(bcoords[0])+""+str(bcoords[1])
        winner = self.map_neurons[bhash]
        score= minkowski(winner.weight_vs,input,2)
        winner.k_coassoc_vs[k] = score
        winner.binarycoassoc_vs[k] = 1
        self.map_neurons[bhash] = winner
        return

    def _process_input(self,input_np_array):
        bmu = self._get_BMU(input_np_array)
        for neu in self.map_neurons.values():
            nhash = str(neu.x_c)+""+str(neu.y_c)

            '''weight adjustment if the neuron is in the neighbourhood of the BMU'''
            if minkowski(bmu.coords().astype(float), neu.coords().astype(float), 2) < self.nr:
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
            nei_coordi = np.array([[(neu.x_c),(int(neu.y_c)-1)], [(neu.x_c),(int(neu.y_c)+1)], [(int(neu.x_c)-1),(neu.y_c)], [(int(neu.x_c)+1),int(neu.y_c)]] )
            p =0
            for coord in nei_coords:
                n=None
                try:
                    n= self.map_neurons[coord]
                    n.res_err+=self.fd*n.res_err

                except KeyError:
                    nwron=neuron(nei_coordi[p][0], nei_coordi[p][1], self.dim)
                    new_weight = 0
                #case a) new node has two consecutive nodes on one of its sides
                #tiroshan and lakmal please implement the code here
                #case b) between two old nodes
                    new_weight_b = self._type_b_weight_init(p,neu)
                    new_weight_a = self._type_a_weight_init(p,neu)
                    new_weight_c = self._type_c_weight_init(p,neu)

                    if new_weight_b.all() ==0:
                        if new_weight_a.all() == 0:
                            if new_weight_c.all() == 0:
                            #print "c==0"
                                new_weight = np.ndarray(shape=(self.dim))
                                new_weight.fill(0.5)
                            else:
                                new_weight = new_weight_c
                        else:
                            new_weight = new_weight_a
                    else:
                        new_weight = new_weight_b

                    nwron.weight_vs = new_weight
                    n=nwron
                self.map_neurons[coord]=n
                p+=1
            bmu.res_err=self.thresh/2
            self.map_neurons[str(bmu.x_c)+""+str(bmu.y_c)]=bmu
        return bmu.coords()

    def _type_b_weight_init(self, side, neuron):
        if side == 0:
            next_watch = str(int(neuron.x_c))+str(int(neuron.y_c-2))
        elif side == 1:
            next_watch = str(int(neuron.x_c))+str(int(neuron.y_c+2))
        elif side == 2:
            next_watch = str(int(neuron.x_c-2))+str(int(neuron.y_c))
        else :
            next_watch = str(int(neuron.x_c+2))+str(int(neuron.y_c))

        try:
            b_type_nwron = self.map_neurons[next_watch]
            new_weight = (b_type_nwron.weight_vs+neuron.weight_vs)/2
        except:
            new_weight = np.array([0])
        return new_weight

    def _type_a_weight_init(self, side, neuron):
        if side == 0:
            next_watch = str(int(neuron.x_c))+str(int(neuron.y_c+1))
        elif side == 1:
            next_watch = str(int(neuron.x_c))+str(int(neuron.y_c-1))
        elif side == 2:
            next_watch = str(int(neuron.x_c+1))+str(int(neuron.y_c))
        else :
            next_watch = str(int(neuron.x_c-1))+str(int(neuron.y_c))

        try:
            a_type_nwron = self.map_neurons[next_watch]
            if np.linalg.norm(a_type_nwron.weight_vs) > np.linalg.norm(neuron.weight_vs):
                new_weight = neuron.weight_vs - (a_type_nwron.weight_vs - neuron.weight_vs)
            else:
                new_weight = neuron.weight_vs + (neuron.weight_vs - a_type_nwron.weight_vs)
        except:
            new_weight =np.array([0])
        return new_weight

    def _type_c_weight_init(self, side, neuron):
        if side == 0 or side == 1:
            next_watch_1 = str(int(neuron.x_c+1))+str(int(neuron.y_c))
            next_watch_2 = str(int(neuron.x_c-1))+str(int(neuron.y_c))
        else:
            next_watch_1 = str(int(neuron.x_c))+str(int(neuron.y_c+1))
            next_watch_2 = str(int(neuron.x_c))+str(int(neuron.y_c-1))

        try:
            c_type_nwron = self.map_neurons[next_watch_1]
            if np.linalg.norm(c_type_nwron.weight_vs) > np.linalg.norm(neuron.weight_vs):
                new_weight = neuron.weight_vs - (c_type_nwron.weight_vs - neuron.weight_vs)
            else:
                new_weight = neuron.weight_vs + (neuron.weight_vs - c_type_nwron.weight_vs)
        except:
            try:
                c_type_nwron = self.map_neurons[next_watch_2]
                if np.linalg.norm(c_type_nwron.weight_vs) > np.linalg.norm(neuron.weight_vs):
                    new_weight = neuron.weight_vs - (c_type_nwron.weight_vs - neuron.weight_vs)
                else:
                    new_weight = neuron.weight_vs + (neuron.weight_vs - c_type_nwron.weight_vs)
            except:
                new_weight=np.array([0])

        return new_weight

    def _classified_inputs(self):
        classified_inputs = []
        for a in self.map_neurons.keys():
            classified_inputs.append(self.map_neurons[a].inputs)
        return  classified_inputs

    '''
def get_gsom(data,spread_factor,dim,nr_s,lr_s,fd,lrr,iterations):

    gsom_map = gsom(spread_factor,dim,nr_s,lr_s,lrr,fd)
    start_time= time.time()
    for j in range(iterations):
        gsom_map.map_sizes.append(len(gsom_map.map_neurons.keys()))
        for i in range(data.shape[0]):
            sys.stdout.write("iteration %d :"%(j+1))
            sys.stdout.write(" : NR = %d: "%(gsom_map.nr))
            sys.stdout.write(" input %d "%(i))
            sys.stdout.write(" map size %d "%(len(gsom_map.map_neurons.keys())))
            sys.stdout.write(" time %d \r"%(time.time()-start_time))
            sys.stdout.flush()
            input = data[i]
            gsom_map._grow_map(input,i)

        gsom_map.nr=gsom_map.nr*(1-gsom_map.lr)
        gsom_map.lr = gsom_map.lr*gsom_map.lr_red_coef*(1-3.85/len(gsom_map.map_neurons.values()))
        if gsom_map.nr <=1 :
            print gsom_map.nr
            return

    return

'''