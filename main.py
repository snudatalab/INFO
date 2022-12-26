"""
Project: Interpreting Workload Classification Model Predictions

Authors:
    - Sooyeon Shim (syshim77@snu.ac.kr)
    - Doyeon Kim (rlaehdus@snu.ac.kr)
    - Jun-Gi Jang (elnino4@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

Affiliation:
    - Data Mining Lab., Seoul National University

File: main.py
     - main file for training and evaluating an interpretable model
     - this code is implemented referring to https://github.com/marcotcr/lime

Version: 1.0.0
"""

import math
import os
# from time import time
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from cluster import cluster_all
from train import train

import random 
from functools import partial
from sklearn import linear_model, metrics

from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

def data_inverse33(data_row, num_samples, sampling_method, cluster_dict):
    """Generates a neighborhood around a prediction.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to
    the means and stds in the training data. For categorical features,
    perturb by sampling according to the training distribution, and making
    a binary feature that is 1 when the value is the same as the instance
    being explained.
    Args:
        data_row: 1d numpy array, corresponding to a row
        num_samples: size of the neighborhood to learn the linear model
        sampling_method: 'gaussian' or 'lhs'
    Returns:
        A tuple (data, inverse), where:
            data: dense num_samples * K matrix, where categorical features
            are encoded with either 0 (not equal to the corresponding value
            in data_row) or 1. The first row is the original instance.
            inverse: same as data, except the categorical features are not
            binary, but categorical (as the original data)
    """

    num_clusters = len(list(set(cluster_dict.keys())))
    num_cols = data_row.shape[0]
    data = np.zeros((num_samples, num_clusters))
    categorical_features = range(num_clusters)
    data = np.array([np.random.randint(0, 2) for _ in range(num_samples * num_clusters)]).reshape(num_samples, num_clusters)

    for index in range(data.shape[0]):
        if np.count_nonzero(data[index,:]) == 0:
            forced = np.random.choice(num_clusters, 1)
            data[index, forced] = 1
        elif np.count_nonzero(data[index,:]) == num_clusters:
            forced = np.random.choice(num_clusters, 1)
            data[index, forced] = 0
    data[0,:] = 1

    inverse = np.zeros((num_samples, num_cols))
    inverse[0,:] = data_row
    for row in (range(num_samples)):
        if row == 0:
            continue
        for col in categorical_features:
            if data[row,col] == 1:
                getcols = cluster_dict[col]
                for ind in getcols:
                    inverse[row,ind] = data_row[ind]
    
    return data, inverse

def kernel(d, kernel_width):
    """
    This is an exponential kernel defined on some distance function.
    Args:
        d: distance function (e.g., euclidean distance)
        kernel_width: kernel width for the exponential kernel
    Returns:
        weights indicating the proximity between samples
    """

    return np.sqrt(np.exp(-(d ** 2) / 1** 2))

def main(X_test, Y_test):
    """
    This is a main function for training and evaluating an interpretable model.
    Args:
        X_test: test dataset
        Y_test: labels of test dataset
    Returns:
        None
    """

    ind = 158449
    interX = X_test[ind]
    interY = int(Y_test[ind])

    num_instances = 10000

    start = time.time()
    print('=== Start making super feature ===')
    cluster_dict = cluster_all()
    print('length of super features: ', len(cluster_dict.keys()))
    test_ngram_zeros = np.load('./final_data/test_ngram_zeros.npy')
    zeros_ind = np.where(test_ngram_zeros[ind, :] == True)[0]
    cluster_dict_ind = {key: [item for item in cluster_dict[key] if item not in zeros_ind] for key in cluster_dict}
    print('=== End making super feature ===\n')

    in_zp, in_z = data_inverse33(interX, num_instances, 'gaussian', cluster_dict_ind)

    scaled_data = in_zp

    distances = metrics.pairwise_distances(
            scaled_data,
            scaled_data[0].reshape(1, -1),
            metric='euclidean'
    ).ravel()

    kernel_width = np.sqrt(X_test.shape[1]) * .75
    kernel_width = float(kernel_width)

    kernel_fn = partial(kernel, kernel_width=kernel_width)
    weights = kernel_fn(distances)

    m = nn.Softmax(dim=1)
    net = torch.load('./models/model_mlp.pt')
    fz = m(net(torch.Tensor(in_z).to(DEVICE)))
    test_fz = fz[0,:]
    fz = np.array(fz.tolist())

    clf2 = linear_model.Ridge(alpha=1, fit_intercept=True)
    clf2.fit(in_zp, fz, sample_weight=weights)

    y_pred_clf2 = clf2.predict(in_zp[0:1, :]).squeeze()
    argsorted_ind = np.argsort(y_pred_clf2)[::-1]
    print(f'=== Classification results of instance {ind} ===')
    print(f'Label of instance: {interY}')
    print(f'Predicted result from f: {int(np.argmax(np.array(test_fz.tolist())))}')
    print(f'Predicted result from g: {argsorted_ind[0]}\n')

    # Check super features and their weights
    sorted_coef = np.argsort(np.abs(clf2.coef_[interY,:]))[::-1]
    print('=== Get weights for interpretation ===')
    print(f'Order of super features: {sorted_coef}')
    print(f'Top-5 weights: {clf2.coef_[interY,sorted_coef[0:5]]}\n')

    # Evaluate top-n accuracy
    print('=== Evaluate the performance ===')
    test_num = 1000
    random_ind = random.sample(list(range(Y_test.shape[0])), test_num)

    count = 0
    check = {}
    hit3 = 0
    for ind in tqdm(random_ind):
        interX = X_test[ind] # 230842
        interY = Y_test[ind]

        num_instances = 10000

        zeros_ind = np.where(test_ngram_zeros[ind, :] == True)[0]
        cluster_dict_ind = {key: [item for item in cluster_dict[key] if item not in zeros_ind] for key in cluster_dict}
        in_zp, in_z = data_inverse33(interX, num_instances, 'gaussian', cluster_dict_ind)

        scaled_data = in_zp

        distances = metrics.pairwise_distances(
                scaled_data,
                scaled_data[0].reshape(1, -1),
                metric='euclidean'
        ).ravel()

        kernel_width = np.sqrt(X_test.shape[1]) * .75
        kernel_width = float(kernel_width)

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        weights = kernel_fn(distances)

        m = nn.Softmax(dim=1)

        fz = m(net(torch.Tensor(in_z).to(DEVICE)))
        test_fz = np.array(fz[0,:].tolist())
        fz = np.array(fz.tolist())

        clf2 = linear_model.Ridge(alpha=1, fit_intercept=True)
        clf2.fit(in_zp, fz, sample_weight=weights)

        y_pred_clf2 = clf2.predict(in_zp[0:1,:]).squeeze()
        argsorted_ind = np.argsort(y_pred_clf2)[::-1]

        if int(np.argmax(test_fz)) == int(argsorted_ind[0]):
            count += 1
        else:
            check[ind] = argsorted_ind[0:2]

        if int(np.argmax(test_fz)) in (argsorted_ind[0:3]):
            hit3 += 1
    print('Top-3 accuracy: ', hit3/test_num)
    print('Top-1 accuracy', count/test_num)

    end = time.time()
    print(f'Run time: {end - start}\n')
    return

if __name__ == '__main__':
    X_test, Y_test = train()
    
    main(X_test, Y_test)
