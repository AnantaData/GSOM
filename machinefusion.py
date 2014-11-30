from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import numpy.random as npr
from gsom import gsomap

__author__ = 'lakmal/laksheen'

class fused_neuron():

    def __init__(self,x_c,y_c,weight):
        self.x_c = x_c
        self.y_c = y_c
        self.weight = weight

def _get_trained_models(gsom, no_samples, features):
    trained_gsoms = []
    idx = npr.randint(0, len(features), (no_samples,int(0.63*len(features))))
    print len(idx[0])
    for i in range(0,no_samples):
        trained_gsoms.append(gsom)
        trained_gsoms[i].process_batch(features[idx[i]],750)
    return trained_gsoms

def _get_coassoc_vec(trained_gsoms):
    coassoc_vec= {}
    i = 1
    for gsom in trained_gsoms:
        for neu in gsom.map_neurons.values():
            coassoc_vec[i,str(neu.coords()[0])+""+str(neu.coords()[1])]=neu
        i+=1
    return coassoc_vec

def _filter_from_usage_threshold_binary_sum(coassoc_vec, usage_threshold):
    for map_coords in coassoc_vec.keys():
        if np.sum(coassoc_vec[map_coords].binarycoassoc_vs) < usage_threshold:
            del coassoc_vec[map_coords]
    return coassoc_vec

def _filter_from_usage_threshold_euclidean_score(coassoc_vec, usage_threshold):
    for map_coords in coassoc_vec.keys():
        if np.linalg.norm(coassoc_vec[map_coords]) > usage_threshold:
            del coassoc_vec[map_coords]
    return coassoc_vec

def _get_incident_matrix_binary(coassoc_vec, fusion_threshold):
    n = len(coassoc_vec.keys())
    incidence_matrix = np.zeros(shape=(n,n))
    for i in range(len(coassoc_vec.keys())-1):
        checkee = coassoc_vec[coassoc_vec.keys()[i+1]]

        if i == len(coassoc_vec.keys())-2:
            incidence_matrix[i+1][i+1] = 1

        incidence_matrix[i][i] = 1

        for j in range(i+1):
            neu = coassoc_vec[coassoc_vec.keys()[j]]
            if jaccard_similarity_score(checkee.binarycoassoc_vs,neu.binarycoassoc_vs) > fusion_threshold :
                incidence_matrix[j][i+1] = 1

    return incidence_matrix

def _get_incident_matrix_cosine_similarity(coassoc_vec, fusion_threshold):
    n = len(coassoc_vec.keys())
    incidence_matrix = np.zeros(shape=(n,n))
    for i in range(len(coassoc_vec.keys())-1):
        checkee = coassoc_vec[coassoc_vec.keys()[i+1]]

        if i == len(coassoc_vec.keys())-2:
            incidence_matrix[i+1][i+1] = 1

        incidence_matrix[i][i] = 1

        for j in range(i+1):
            neu = coassoc_vec[coassoc_vec.keys()[j]]
            if cosine_similarity(checkee.coassoc_vs,neu.coassoc_vs) > fusion_threshold :
                incidence_matrix[j][i+1] = 1

    return incidence_matrix

def _group_codebooks(incidence_matrix,codebooklen):
    groups = np.zeros(codebooklen)
    group_num=0
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
    return groups


def _get_fused_gsom(gsom,coassoc_vec, codebook_groups, connection_threshold):
    group_neurons = []
    for j in range(int(max(codebook_groups))):
        group_neurons.append([])

    for i in range(len(codebook_groups)):
        group_neurons[int(codebook_groups[i])-1].append(i)

    incidence_ensemble = _get_incidence_ensemble(group_neurons,coassoc_vec)
    new_neurons =[]
    for group in group_neurons:
        weight_sum = 0
        x_co_sum  = 0
        y_co_sum = 0
        n = len(group)
        for val in group:
            neu = coassoc_vec[coassoc_vec.keys()[val]]
            weight_sum+= neu.weight_vs
            x_co_sum += neu.x_c
            y_co_sum += neu.y_c
        new_neurons.append(fused_neuron(x_co_sum/n,y_co_sum/n,weight_sum/n))

    for k in range(incidence_ensemble.shape[0]):
        for l in range(k,incidence_ensemble.shape[0]):
            if(incidence_ensemble[k][l]>connection_threshold):
                if(k!=l):
                    print "connected",k,"and",l , "x1 ",new_neurons[k].x_c, " y1 ", new_neurons[k].y_c, "x2 ",new_neurons[l].x_c, " y2 ", new_neurons[l].y_c
    print incidence_ensemble
    fused_gsom = gsom
    fused_gsom.create_fused_gsom(new_neurons)
    return fused_gsom

def _get_incidence_ensemble(group_neurons,coassoc_vec):
    n = len(group_neurons)
    incidence_ensemble = np.zeros(shape=(n,n))
    for i in range(len(group_neurons)):
        incidence_ensemble[i][i] = 1
        for j in range(i,len(group_neurons)):
            incidence_ensemble[i][j]= _get_max_similarity(group_neurons[i],group_neurons[j],coassoc_vec)
    return incidence_ensemble


def _get_max_similarity(list1,list2, coassoc_vec):
    n = len(coassoc_vec.keys())
    max = 0
    for i in range(len(list1)):
        checkee = coassoc_vec[coassoc_vec.keys()[list1[i]]]
        for j in range(len(list2)):
            neu = coassoc_vec[coassoc_vec.keys()[list2[j]]]
            jaccard = jaccard_similarity_score(checkee.binarycoassoc_vs,neu.binarycoassoc_vs)
            if jaccard> max:
                max = jaccard
    return max

def machine_fusion_gsom(data, gsom, no_samples, usage_threshold, fusion_threshold,connection_threshold):
    trained_gsoms       = _get_trained_models(gsom,no_samples,data)
    coassoc_vec         = _get_coassoc_vec(trained_gsoms)
    coassoc_vec         = _filter_from_usage_threshold_binary_sum(coassoc_vec,usage_threshold)
    coassoc_vec         = _filter_from_usage_threshold_euclidean_score(coassoc_vec,usage_threshold)
    incident_mat        = _get_incident_matrix_binary(coassoc_vec,fusion_threshold)
    incident_mat        = _get_incident_matrix_cosine_similarity(coassoc_vec,fusion_threshold)
    print incident_mat
    codebook_groups     = _group_codebooks(incident_mat,len(coassoc_vec.keys()))
    print codebook_groups
    return _get_fused_gsom(gsom, coassoc_vec,codebook_groups, connection_threshold)

