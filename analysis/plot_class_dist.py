import os
import os.path as osp
from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 16})

import numpy as np
import pandas as pd
import seaborn as sns
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to 70company-like dataset")
    parser.add_argument(
        "--outfile",
        help="Path to 70company-like dataset",
        default="./figures/current-output-debug.png",
    )

    args = parser.parse_args()
    data = pd.DataFrame(
        {
            "year": np.load(osp.join(args.path, "t.npy")),
            "label": np.load(osp.join(args.path, "y.npy")),
        }
    )

    print("Label v