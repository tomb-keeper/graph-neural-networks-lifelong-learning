import itertools as it
import os.path as osp
import tempfile

import dgl
from dgl import DGLGraph, DGLError
from dgl.data.utils import load_graphs
import networkx as nx
import numpy as np
import pandas as pd
import torch
from joblib import Memory
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import os
import gc

# Globals
CACHE_DIR = "./cache"
MEMORY = Memory(CACHE_DIR, verbose=2)


def make_geometric_dataset(edge_index, features, labels, edge_attr=None):
    # One data object is one graph
    import torch_geometric as tg

    data = tg.data.Data(
        x=features, edge_index=edge_index, edge_attr=edge_attr, y=labels
    )
    # Just as in regular PyTorch, you do not have to use datasets, e.g., when
    # you want to create synthetic data on the fly without saving them
    # explicitly to disk. In this case, simply pass a regular python list
    # holding torch_geometric.data.Data objects
    # Source:
    # https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
    return [data]


@MEMORY.cache
def load_data(path, backend="dgl", format="tuple"):
    if backend == "dgl":
        try:
            print("Trying to load dgl graph directly")
            glist, __ = load_graphs(osp.join(path, "g.bin"))
            g = glist[0]
            print("Success")
        except DGLError as e:
            print("File not found", e)
            print("Loading nx graph")
            nx_graph = nx.read_adjlist(osp.join(path, "adjlist.txt"), nodetype=int)
            print("Type:", type(nx_graph))
            g = dgl.from_networkx(nx_graph)
        N = g.number_of_nodes()
        X = np.load(osp.join(path, "X.npy"))
        y = np.load(osp.join(path, "y.npy"))
        t = np.load(osp.join(path, "t.npy"))
        assert X.shape[0] == N
        assert y.size == N
        assert t.size == N
        return g, X, y, t
    elif backend == "geometric":
        import torch_geometric as tg

        # DONE test this!
        nx_graph = nx.read_adjlist(osp.join(path, "adjlist.txt"), nodetype=int)
        X = np.load(osp.join(path, "X.npy"))
        y = np.load(osp.join(path, "y.npy"))
        t = np.load(osp.join(path, "t.npy"))
        print("Type:", type(nx_graph))
        attr_dict = {i: {"X": X[i], "y": y[i], "t": t[i]} for i in range(X.shape[0])}
        print("attr_dict loaded!")
        nx.set_node_attributes(nx_graph, attr_dict)
        print("attributes set!")
        del attr_dict
        gc.collect()
        g = tg.utils.from_networkx(nx_graph)
        del nx_graph
        if format == "tuple":
            return g.edge_index, g.X, g.y, g.t
        else:
