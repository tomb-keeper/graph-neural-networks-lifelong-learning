import os
import os.path as osp
from datasets import load_data
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})
import numpy as np
import pandas as pd
import argparse
from collections import Counter
from operator import itemgetter
import dgl
from dgl.data import register_data_args, load_data as load_data_dgi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to 70company-like dataset")
    parser.add_argument(
        "--outfile",
        help="Path to 70company-like dataset",
        default="./figures/degree.png",
    )
    parser.add_argument(
        "--dmin",
        help="Custom min degree for computing power law exp, defaults to min value from data",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--remove-self-loops",
        help="Remove self loops",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    try:
        g, _, _, _ = load_data(args.path)
    except FileNotFoundError:
        print("Trying to load", args.path, "via DGL")
        g = dgl.DGLGraph(load_data_dgi(argparse.Namespace(dataset=args.path)).graph)
    print(g)
    if args.remove_self_loops:
        print("Removing 