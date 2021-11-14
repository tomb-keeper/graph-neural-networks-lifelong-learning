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
from 