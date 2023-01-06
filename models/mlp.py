""" A simple yet generic MLP """
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes,
                 n_layers, activation, dropout):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
   