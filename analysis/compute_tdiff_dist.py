import argparse
import numpy as np
import pandas as pd
import dgl
from tqdm import tqdm

from datasets import load_data

parser = argparse.ArgumentParser()
parser.add_argument('data_path')
parser.add_argument('--max-hops', type=int, default=2)
parser.add_argument('--save', default=None)
args = parser.parse_args()

g, __, __, ts = load_d