"""
GCN implementation of the DGL library (https://dgl.ai) with minor modifications
to facilitate dynamically changing graph structure.

Source: https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/gcn_mp.py
"""
import math
import torch
import torch.nn as nn

# In DGL, GraphConv refers to Kipf's GCN Conv operator
from dgl.nn.pytorch import 