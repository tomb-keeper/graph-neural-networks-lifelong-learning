import os.path as osp
import json
import argparse
import numpy as np
import pandas as pd

def resolve_classes(class_set, index2class):
    return [index2class[c] for c in list(class_set)]


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('path', help="Path to graph dataset")
    parser.add_argument('time', help="Path to numpy data file with task ids")
    parser.add_argument('labels', help="Path to numpy data file with labels")
    parser.add_argument('--label2index', help="Path to json file containing the mapping from label to index")
    parser.add_argument('--start', type=int, help="Start year")
    parser.add_argument('--end', type=int, help="Start year")

    args = parser.parse_args()
    data = pd.DataFrame(
        {
          # 'year': np.load(osp.join(args.path, 't.npy')),
         # 'label': np.load(osp.join(args.path, 'y.npy'))
          'year': np.load(args.time),
         'label': np.load(args.labels)
        }
    )
    t_start = args.start if args.start is not None else data.year.min()
    t_end = args.end if args.end is not None else data.year.max()

    if args.label2index is not None:

        with open(osp.join(args.path, 'label2index.json'), 'r') as fh:
            label2index = json.load(fh)

    