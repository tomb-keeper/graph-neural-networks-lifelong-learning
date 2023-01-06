""" A simple yet generic MLP """
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes,
            