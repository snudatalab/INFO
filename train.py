"""
Project: Interpreting Workload Classification Model Predictions

Authors:
    - Sooyeon Shim (syshim77@snu.ac.kr)
    - Doyeon Kim (rlaehdus@snu.ac.kr)
    - Jun-Gi Jang (elnino4@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

Affiliation:
    - Data Mining Lab., Seoul National University

File: train.py
     - implementation for training a workload classification model

Version: 1.0.0
"""

import math
import os
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dataloader import concatenate_all, concatenate_ngram_vecs
from model import FNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

class WorkloadDataset(Dataset):
    """
    This class represents dataloader for workload subsequences
    """

    def __init__(self, numpy_data, numpy_dataY):
        """
        Initialize the class.
    
        :param numpy_data: the concatenation of n-gram feature vectors, address-related feature vectors
        :param numpy_dataY: labels corresponding to feature vectors
        """
        self.data = np.concatenate((numpy_data,  numpy_dataY[:,np.newaxis]), axis=1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

def train():
    """
    This function trains and evaluates a workload classification model.
    Args:
        None
    Returns:
        A tuple (X_test, Y_test), where:
            X_test: test dataset
            Y_test: labels of test dataset
    """

    batch_size  = 64
    num_classes = 31

    data = './final_data/'
    ids_path = os.path.join(data, 'data_split_ids/')
    path_7gram = os.path.join(data, '7-grams/')
    path_11gram = os.path.join(data, '11-grams/')
    path_15gram = os.path.join(data, '15-grams/')

    mypath_bank = os.path.join(data, 'bank_access_counts/')
    mypath_address = os.path.join(data, 'row_col_address_access_counts/')

    _, _, X_test_ngram, _, _, _ = concatenate_ngram_vecs(ids_path, path_7gram, path_11gram, path_15gram, num_classes)
    with open('./final_data/test_ngram_zeros.npy', 'wb') as f:
        np.save(f, X_test_ngram == 0)

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = concatenate_all(ids_path, path_7gram, path_11gram, path_15gram, mypath_bank, mypath_address, num_classes)
    trn_loader = DataLoader(WorkloadDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    vld_loader = DataLoader(WorkloadDataset(X_valid, Y_valid), batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(WorkloadDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

    input_size = X_train.shape[1]
    learning_rate = 0.0001
    net = FNN(input_size, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    print('Training a classification model...')

    best_vld_loss = math.inf
    for epoch in range(1):
        t1 = time()
        trn_loss, vld_loss, test_loss = 0., 0., 0.
        trn_hit, vld_hit, test_hit = 0, 0, 0
        trn_total, vld_total, test_total = 0, 0, 0

        net.train()
        for i, batch in enumerate(trn_loader, 0):
            x, y = batch[:, :-1], batch[:, -1]
            x = x.type(torch.FloatTensor).to(DEVICE)
            y = y.type(torch.LongTensor).to(DEVICE)
            optimizer.zero_grad()
            pred = net(x)

            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            trn_loss += loss.item()

            pred_cls = torch.argmax(pred, dim=1)
            trn_hit += sum(pred_cls == y)
            trn_total += y.shape[0]

        trn_loss /= (i + 1)
        trn_acc = float(trn_hit) / trn_total

        net.eval()
        for i, batch in enumerate(vld_loader, 0):
            x, y = batch[:, :-1], batch[:, -1]
            x = x.type(torch.FloatTensor).to(DEVICE)
            y = y.type(torch.LongTensor).to(DEVICE)
            pred = net(x)
            loss = criterion(pred, y)
            vld_loss += loss.item()
            pred_cls = torch.argmax(pred, dim=1)
            vld_hit += sum(pred_cls == y)
            vld_total += y.shape[0]

        vld_loss /= (i + 1)
        vld_acc = float(vld_hit) / vld_total

        for i, batch in enumerate(test_loader, 0):
            x, y = batch[:, :-1], batch[:, -1]
            x = x.type(torch.FloatTensor).to(DEVICE)
            y = y.type(torch.LongTensor).to(DEVICE)
            pred = net(x)
            loss = criterion(pred, y)
            test_loss += loss.item()
            pred_cls = torch.argmax(pred, dim=1)
            test_hit += sum(pred_cls == y)
            test_total += y.shape[0]

        test_loss /= (i + 1)
        test_acc = float(test_hit) / test_total

        if vld_loss < best_vld_loss:
            best_vld_loss = vld_loss
            best_result = f'[BEST] Epoch: {epoch + 1:03d}, TrnLoss: {trn_loss:.4f}, VldLoss: {vld_loss:.4f}, TestLoss: {test_loss:.4f}, ' \
                        f'TrnAcc: {trn_acc:.4f}, VldAcc: {vld_acc:.4f}, TestAcc: {test_acc:.4f}'
        print(
            f'[{time() - t1:.2f}sec] Epoch: {epoch + 1:03d}, TrnLoss: {trn_loss:.4f}, VldLoss: {vld_loss:.4f}, TestLoss: {test_loss:.4f}, '
            f'TrnAcc: {trn_acc:.4f}, VldAcc: {vld_acc:.4f}, TestAcc: {test_acc:.4f}')
        print(best_result)

    torch.save(net, './models/model_mlp.pt')
    print('Finish to train a classification model')
    
    return X_test, Y_test
    
if __name__ == '__main__':
    train()
