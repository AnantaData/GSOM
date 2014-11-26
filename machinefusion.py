from sklearn.cross_validation import train_test_split

__author__ = 'lakmal/laksheen'

#from sklearn import cross_validation
from sklearn.utils import resample
#data - the data set , gsom - the gsom to be  used

def _get_bootstrap_samples(data):
    bootstrap_data = []
    #bs = cross_validation.Bootstrap(9, random_state=0)
    return bootstrap_data

def _get_trained_models(gsom, no_samples, features):
    trained_gsoms = []
    #bs = cross_validation.Bootstrap(no_samples, random_state=0)
    #bs = resample(features, n_samples=no_samples, random_state=0)
    #print bs
    i=0
    for train_index, test_index in range(0,no_samples):
        features_train, features_test = train_test_split(features, test_size=0.33, random_state=42)
        trained_gsoms.append(gsom)
        trained_gsoms[i].process_batch(features[train_index],750)
    return trained_gsoms

def _get_coassoc_vec(trained_gsoms):
    coassoc_vec= {}
    return coassoc_vec

def _filter_from_usage_threshold(coassoc_vec):
    return coassoc_vec

def _get_incident_matrix(coassoc_vec):
    incident_mat = []
    return incident_mat

def _group_codebooks(incident_mat):
    codebook_groups = {}
    return codebook_groups


def _get_fused_gsom(codebook_groups):
    fused_gsom = None
    return fused_gsom


def machine_fusion_gsom(data, gsom):
    bootstrap_data      = _get_bootstrap_samples(data)
    trained_gsoms       = _get_trained_models()
    coassoc_vec         = _get_coassoc_vec(trained_gsoms)
    coassoc_vec         = _filter_from_usage_threshold(coassoc_vec)
    incident_mat        = _get_incident_matrix(coassoc_vec)
    codebook_groups     = _group_codebooks(incident_mat)
    return _get_fused_gsom(codebook_groups)


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

gsom = gsomap(SP=0.9,dims=16,nr_s=4,lr_s=0.9,fd=0.99999)

print _get_trained_models(gsom, 10,features)