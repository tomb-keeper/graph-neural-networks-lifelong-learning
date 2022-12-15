"""
GCN implementation of the DGL library (https://dgl.ai) with minor modifications
to facilitate dynamically changing graph structure.

Source: https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/gcn_mp.py
"""
import math
import torch
import torch.nn as nn

# In DGL, GraphConv refers to Kipf's GCN Conv operator
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 improved=True):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in ra