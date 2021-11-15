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
    # explici