import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json
import argparse
import os.path

argparser = argparse.ArgumentParser(description="Process PR curves")

argparser.add_argument('-s', '--show', action='store_true', help='Display figures as windows')
argparser.add_argument('-w', '--write', action='store_true', help='Write figures to files')
argparser.add_argument('-u', '--update', action='store_true', help='Force update csv file')

datafilejson = 'sweep.json'
datafilecsv = 'sweep.csv'

def main(args):
    figs = []

    jsonTime = os.path.getmtime(datafilejson)
    csvTime = os.path.getmtime(datafilecsv)
    if jsonTime > csvTime or args.update:
        with open(datafilejson) as b:
            raw = json.load(b)

        def transform(x):
            d = pd.io.json.json_normalize(x).transpose()
            d = pd.Series(d[0])
            return d

        d = pd.io.json.json_normalize(raw['curves'], record_path=['data'], meta=['parameters', 'times'])
        d = pd.concat([d, d['parameters'].apply(transform), d['times'].apply(transform)], axis=1)
        d = d.drop(['parameters', 'times'], axis=1)
        d.to_csv(datafilecsv, index=False)
    else:
        d = pd.read_csv(datafilecsv)

    nu = d.nunique(axis=0)
    d = d.drop(nu[nu == 1].index, axis=1)
    d['Time'] = d['Difference matrix calculation'] + d['Sequence search'] + d['Difference matrix enhancement']
    print(d.columns)

    idx = d.groupby(['Number of trajectories', 'Sequence length'])['F1 Score'].transform(max) == d['F1 Score']
    d = d[idx]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(d['Sequence length'], d['Number of trajectories'], d['Time'])

    if args.show:
        plt.show()

    if args.write:
        for fig, name in figs:
            fig.savefig(prefix + '-' + name + '.pdf')

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
