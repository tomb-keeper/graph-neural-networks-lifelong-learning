
import argparse
import pandas as pd
pd.set_option('display.max_rows', None)
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("results_csv_file", help="Path to results file", nargs='+')
parser.add_argument("--latex", default=False, action='store_true', help="Produce latex output")
parser.add_argument("--save", help="Path to save resulting table")

args = parser.parse_args()
print("Loading data:", args.results_csv_file)
df = pd.read_csv(args.results_csv_file[0])
print("N =",len(df))
for path in args.results_csv_file[1:]:
    print("Adding data:", path)
    add_data = pd.read_csv(path)
    df = pd.concat([df, add_data], axis=0, ignore_index=True)
    print("N =",len(df))

def SD(values):
    """ Just to ensure we use ddof=1 """
    return values.std(ddof=1)

def SE(values):
    return values.std(ddof=1) / np.sqrt(values.count())

def forward_transfer(df):
    print("Computing forward transfer")
    first_task_per_dataset = df.groupby('dataset', as_index=False)['year'].min()
    print("Dropping first task per dataset:", first_task_per_dataset, sep='\n')
    for dataset, first_task in first_task_per_dataset.itertuples(index=False):
        idx = df[(df['dataset'] == dataset) & (df['year'] == first_task)].index
        df = df.drop(idx, axis=0)

    df = df.pivot_table(index=['dataset', 'model', 'history', 'year'], columns=['start'], aggfunc='mean')
    df['FWT'] = df['accuracy']['warm'] - df['accuracy']['cold']
    df.drop('accuracy', axis=1, inplace=True)
    # Average out the tasks (years)
    fwt = df.groupby(['dataset','model','history']).FWT.mean()
    return fwt