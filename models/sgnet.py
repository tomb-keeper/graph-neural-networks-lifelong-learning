""" Subclass of DGL's Simplified GCN Implementation to enable incremental training"""

from dgl.nn.pytorch.conv.sgconv import SGConv


class SGNet(SGConv):
    def __reset_ca