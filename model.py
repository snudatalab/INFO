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

File: model.py
     - implementation of constructing a workload classification model (2-layer MLP)

Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FNN(nn.Module):
    """
    This class represents an MLP model for workload subsequences
    """

    def __init__(self, input_size, num_classes):
        """
        Initialize the class
    
        :param input_size: size of an input feature vector
        :param num_classes: number of known classes
        """
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, num_classes)

        self.bn1 = nn.BatchNorm1d(100)

    def forward(self, x):
        """
        Forward the layer given data batch

        :param x: data batch
        :return: forwarded results
        """
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.fc2(out)
        return out

