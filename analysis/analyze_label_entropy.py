import os
import os.path as osp

import numpy as np
import pandas as pd
import argparse

import numpy as np
import scipy.stats as stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help="Path to 70company-like dataset")

    args = parser.parse_args()
    data = pd.DataFrame(
        {