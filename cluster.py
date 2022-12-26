"""
Project: Interpreting Workload Classification Model Predictions

Authors:
    - Sooyeon Shim (syshim77@snu.ac.kr)
    - Doyeon Kim (rlaehdus@snu.ac.kr)
    - Jun-Gi Jang (elnino4@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

Affiliation:
    - Data Mining Lab., Seoul National University

File: cluster.py
     - implementation of generating super features from original features by clustering

Version: 1.0.0
"""

import pickle
from tqdm import tqdm
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def cluster_all():
    """
    This function clusters the original features and generates super features for an interpretable model.
    Args:
        None
    Returns:
        clustering results which are CMD, bank-level, and cell-level super features
    """

    # Load n-grams information
    path = './sequences'
    with open(path + '/11-grams.top25_osr.pkl', 'rb') as f:
        data11 = pickle.load(f)
        
    with open(path + '/7-grams.top25_osr.pkl', 'rb') as f:
        data7 = pickle.load(f)
        
    with open(path + '/15-grams.top25_osr.pkl', 'rb') as f:
        data15 = pickle.load(f)

    feature_names = list(data7.keys()) + list(data11.keys()) + list(data15.keys())

    feature_names_dic = {}
    for i in range(len(feature_names)):
        feature_names_dic[feature_names[i]] = [i]
    
    # clustering for n-gram    
    similarity_matrix = np.zeros((len(feature_names), len(feature_names)))

    for i in tqdm(range(len(feature_names))):
        for j in range(len(feature_names)):
            # s1 end, s2 front
            len1 = len(feature_names[i])
            len2 = len(feature_names[j])
            if len1 > len2:
                seq1 = feature_names[i]
                seq2 = feature_names[j]
            else:
                seq1 = feature_names[j]
                seq2 = feature_names[i]
            pos = 0
            min_len = min(len1, len2)
            for k in range(len(seq2)):
                if seq1[-(min_len-k):] == seq2[:min_len-k]:
                    pos = min_len-k
                    break
                    
            similarity_first = (pos)/(len(feature_names[i]) + len(feature_names[j]) - pos)
                    
            # s1 end, s2 front
            len1 = len(feature_names[i])
            len2 = len(feature_names[j])
            if len1 > len2:
                seq1 = feature_names[i]
                seq2 = feature_names[j]
            else:
                seq1 = feature_names[j]
                seq2 = feature_names[i]
            pos = 0
            min_len = min(len1, len2)
            for k in range(len(seq2)):
                if seq1[:(min_len-k)] == seq2[-(min_len-k):]:
                    pos = min_len-k
                    break
                    
            similarity_second = (pos)/(len(feature_names[i]) + len(feature_names[j]) - pos)
            similarity_matrix[i][j] = max(similarity_first, similarity_second)
            
    hierarchical_cluster = AgglomerativeClustering(n_clusters=6, affinity='precomputed', linkage='average', compute_distances=True)
    clustering = hierarchical_cluster.fit(1-similarity_matrix)
    cluster_dict = {}
    for i in range(clustering.labels_.shape[0]):
        if clustering.labels_[i] in list(cluster_dict.keys()):
            cluster_dict[clustering.labels_[i]].append(i)
        else:
            cluster_dict[clustering.labels_[i]] = [i]

    # clustering for bank and address
    bank_start_ind = max(set(cluster_dict.keys())) + 1
    cluster_dict[bank_start_ind] = []
    cluster_dict[bank_start_ind+1] = []
    for i in range(543, 559, 1):
        cluster_dict[bank_start_ind].append(i)
    for i in range(559, 575, 1):
        cluster_dict[bank_start_ind+1].append(i)
    
    address_start_ind = max(set(cluster_dict.keys())) + 1
    for i in range(8):
        cluster_dict[i+address_start_ind] = []
    add_st = 543+32
    for i in range(1024):
        ind_tmp = int(i // 128)
        cluster_dict[address_start_ind+ind_tmp].append(i+add_st)
    
    return cluster_dict
