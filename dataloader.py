#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project: Interpreting Workload Classification Model Predictions

Authors:
    - Sooyeon Shim (syshim77@snu.ac.kr)
    - Doyeon Kim (rlaehdus@snu.ac.kr)
    - Jun-Gi Jang (elnino4@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

Affiliation:
    - Data Mining Lab., Seoul National University

File: dataloader.py
     - dataloader file for loading feature vectors of subsequences

Version: 1.0.0
"""

from os import listdir
from os.path import isfile, join
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from collections import Counter


def concatenate_all(ids_path, path_7gram, path_11gram, path_15gram, path_bank, path_address, num_known_classes):
    """
    This function loads n-gram CMD vectors, address-related vectors, and time vectors and concatenate them

    :param ids_path: data index path to split data into train, valid, and test
    :param path_7gram: 7gram vector path
    :param path_11gram: 11gram vector path
    :param path_15gram: 15gram vector path
    :param path_bank: bank counting vector path
    :param path_address: address counting vector path
    :param num_known_classes: number of known classes
    :rtype tuples of numpy arrays
    :return: the concatenated address-related vectors
    """

    X1_train, X1_valid, X1_test, Y_train, Y_valid, Y_test = read_ngram_vec(ids_path, path_7gram, num_known_classes)
    X2_train, X2_valid, X2_test, _, _, _ = read_ngram_vec(ids_path, path_11gram, num_known_classes)
    X3_train, X3_valid, X3_test, _, _, _ = read_ngram_vec(ids_path, path_15gram, num_known_classes)
    X_train_bank, X_valid_bank, X_test_bank = read_bank_vec(ids_path, path_bank, num_known_classes)
    X_train_address, X_valid_address, X_test_address = read_address_vec(ids_path, path_address, num_known_classes)

    X_train = np.concatenate([X1_train,X2_train,X3_train,X_train_bank,X_train_address], axis=1)
    X_valid = np.concatenate([X1_valid,X2_valid,X3_valid,X_valid_bank,X_valid_address], axis=1)
    X_test = np.concatenate([X1_test,X2_test,X3_test,X_test_bank,X_test_address], axis=1)

    # Standardization
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


def concatenate_ngram_vecs(ids_path, path_7gram, path_11gram, path_15gram, num_known_classes):
    """
    This function loads n-gram CMD vectors and concatenate them

    :param ids_path: data index path to split data into train, valid, and test
    :param path_7gram: 7gram vector path
    :param path_11gram: 11gram vector path
    :param path_15gram: 15gram vector path
    :param num_known_classes: number of known classes
    :rtype tuples of numpy arrays
    :return: the concatenated CMD vectors and y of train, valid, test data
    """
    X1_train, X1_valid, X1_test, Y_train, Y_valid, Y_test = read_ngram_vec(ids_path, path_7gram, num_known_classes)

    X2_train, X2_valid, X2_test, _, _, _ = read_ngram_vec(ids_path, path_11gram, num_known_classes)
    X3_train, X3_valid, X3_test, _, _, _ = read_ngram_vec(ids_path, path_15gram, num_known_classes)

    X_train = np.concatenate([X1_train, X2_train, X3_train], axis=1)
    X_valid = np.concatenate([X1_valid, X2_valid, X3_valid], axis=1)
    X_test = np.concatenate([X1_test, X2_test, X3_test], axis=1)

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


def train_test_idx(mypath):
    """
    This function loads indices to split data into train, valid, and test

    :param mypath: split indices path
    :rtype tuples of dictionaries
    :return: indices of train, valid, test data
    """
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = sorted(onlyfiles)

    split_train_ids_dic = {}
    split_valid_ids_dic = {}
    split_test_ids_dic = {}
    for i in range(len(onlyfiles)):
        path = mypath + onlyfiles[i]
        with open(path, 'rb') as f:
            (train_ids, val_ids, test_ids) = pickle.load(f)
            split_train_ids_dic[i] = train_ids
            split_valid_ids_dic[i] = val_ids
            split_test_ids_dic[i] = test_ids 

    return split_train_ids_dic, split_valid_ids_dic, split_test_ids_dic


def read_ngram_vec(ids_path, mypath, num_known_classes):
    """
    This function loads n-gram CMD vector

    :param ids_path: data index path to split data into train, valid, and test
    :param mypath: n-gram CMD vector path
    :param num_known_classes: number of known classes
    :rtype tuples of numpy arrays 
    :return: n-gram CMD vectors and y of train, valid, test data
    """

    idx_path = ids_path
    split_train_ids_dic, split_valid_ids_dic, split_test_ids_dic = train_test_idx(idx_path)

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = sorted(onlyfiles)

    X1_train = []
    X1_valid = []
    X1_test = []

    Y_train = []
    Y_valid = []
    Y_test = []

    for i, file_name in tqdm(enumerate(onlyfiles)):
        path = mypath + file_name
        with open(path, 'rb') as f:
            X_tmp = np.array(pickle.load(f)).squeeze()
            if i < num_known_classes:
                X1_train.append(X_tmp[split_train_ids_dic[i],:])
                X1_valid.append(X_tmp[split_valid_ids_dic[i],:])
                X1_test.append(X_tmp[split_test_ids_dic[i],:])        
                length1 = len(split_train_ids_dic[i])
                length2 =len(split_valid_ids_dic[i])
                length3 = len(split_test_ids_dic[i])
                Y_tmp1 = np.ones((length1,)) * i
                Y_tmp2 = np.ones((length2,)) * i
                Y_tmp3 = np.ones((length3,)) * i
                Y_train.append(Y_tmp1)
                Y_valid.append(Y_tmp2)
                Y_test.append(Y_tmp3)

    X1_train = np.concatenate(X1_train, axis=0)
    X1_valid = np.concatenate(X1_valid, axis=0)
    X1_test = np.concatenate(X1_test, axis=0)

    Y_train = np.concatenate(Y_train, axis=0)
    Y_valid = np.concatenate(Y_valid, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)

    return X1_train, X1_valid, X1_test, Y_train, Y_valid, Y_test


def read_bank_vec(ids_path, mypath, num_known_classes):
    """
    This function loads bank counting vectors

    :param ids_path: data index path to split data into train, valid, and test
    :param mypath: bank counting vector path
    :param num_known_classes: number of known classes
    :rtype tuples of numpy arrays 
    :return: bank counting vectors
    """

    idx_path = ids_path
    split_train_ids_dic, split_valid_ids_dic, split_test_ids_dic = train_test_idx(idx_path)

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = sorted(onlyfiles)

    X0_train = []
    X0_valid = []
    X0_test = []

    for i, file_name in tqdm(enumerate(onlyfiles)):
        path = mypath + file_name     
        with open(path, 'rb') as f:
            X_tmp = np.array(pickle.load(f)).squeeze()            
            if i<num_known_classes:
                X0_train.append(X_tmp[split_train_ids_dic[i],:])
                X0_valid.append(X_tmp[split_valid_ids_dic[i],:])
                X0_test.append(X_tmp[split_test_ids_dic[i],:])

    X0_train = np.concatenate(X0_train, axis=0)
    X0_valid = np.concatenate(X0_valid, axis=0)
    X0_test = np.concatenate(X0_test, axis=0)

    return X0_train, X0_valid, X0_test


def read_address_vec(ids_path, mypath, num_known_classes):
    """
    This function loads address counting vectors

    :param ids_path: data index path to split data into train, valid, and test
    :param mypath: bank counting vector path
    :param num_known_classes: number of known classes
    :rtype tuples of numpy arrays 
    :return: address counting vectors
    """

    idx_path = ids_path
    split_train_ids_dic, split_valid_ids_dic, split_test_ids_dic = train_test_idx(idx_path)

    X_train_address = []
    X_valid_address = []
    X_test_address = []

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = sorted(onlyfiles)

    for i, name in tqdm(enumerate(onlyfiles)):
        filename = name
        loadname = mypath+filename
        address = np.load(loadname,allow_pickle=True)
        if i< num_known_classes:
            X_train_address.append(address[split_train_ids_dic[i],:])
            X_valid_address.append(address[split_valid_ids_dic[i],:])
            X_test_address.append(address[split_test_ids_dic[i],:])

    X_train_address = np.concatenate(X_train_address, axis=0)
    X_valid_address = np.concatenate(X_valid_address, axis=0)
    X_test_address = np.concatenate(X_test_address, axis=0)    

    return X_train_address, X_valid_address, X_test_address
