__author__ = 'lakmal/laksheen'

#data - the data set , gsom - the gsom to be  used

def _get_bootstrap_samples(data):
    bootstrap_data = []
    return bootstrap_data

def _get_trained_models(gsom, bootstrap_data):
    trained_gsoms = []
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

