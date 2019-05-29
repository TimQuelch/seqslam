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

datafileJson = 'sweep.json'
datafileCsv = 'sweep.csv'

def checkUpdateRequired(original, processed):
    if not os.path.isfile(processed):
        return True

    oTime = os.path.getmtime(original)
    pTime = os.path.getmtime(processed)

    if oTime > pTime:
        return True

    return False


def main(args):
    figs = []

    if checkUpdateRequired(datafileJson, datafileCsv) or args.update:
        with open(datafileJson) as b:
            raw = json.load(b)

        def transform(x):
            d = pd.io.json.json_normalize(x).transpose()
            d = pd.Series(d[0])
            return d

        d = pd.io.json.json_normalize(raw['curves'], record_path=['data'], meta=['parameters', 'times'])
        d = pd.concat([d, d['parameters'].apply(transform), d['times'].apply(transform)], axis=1)
        d = d.drop(['parameters', 'times'], axis=1)
        d.to_csv(datafileCsv, index=False)
    else:
        d = pd.read_csv(datafileCsv)

    nu = d.nunique(axis=0)
    d = d.drop(nu[nu == 1].index, axis=1)
    d['Time'] = d['Difference matrix calculation'] + d['Sequence search'] + d['Difference matrix enhancement']

    idx = d.groupby(['Patch window size', 'Sequence length'])['F1 Score'].transform(max) == d['F1 Score']
    d = d[idx]

    timeGrid = d[['Sequence length', 'Patch window size', 'Difference matrix enhancement']]
    timeGrid = timeGrid.set_index(['Sequence length', 'Patch window size']).sort_index()
    timeGrid = timeGrid.unstack(level='Patch window size')

    f1Grid = d[['Sequence length', 'Patch window size', 'Iterations']]
    f1Grid = f1Grid.set_index(['Sequence length', 'Patch window size']).sort_index()
    f1Grid = f1Grid.unstack(level='Patch window size')

    X, Y = np.meshgrid(d['Sequence length'].unique(), d['Patch window size'].unique())
    print(X)
    print(Y)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, timeGrid)
    ax.set_xlabel('ws')
    ax.set_ylabel('sl')
    ax.set_zlabel('Time')

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, f1Grid)
    ax.set_xlabel('ws')
    ax.set_ylabel('sl')
    ax.set_zlabel('F1')

    fig, ax = plt.subplots()
    ax.contourf(X, Y, timeGrid, legend=True)
    ax.set_xlabel('ws')
    ax.set_ylabel('sl')

    fig, ax = plt.subplots()
    ax.contourf(X, Y, f1Grid, legend=True)
    ax.set_xlabel('ws')
    ax.set_ylabel('sl')

    if args.show:
        plt.show()

    if args.write:
        for fig, name in figs:
            fig.savefig(prefix + '-' + name + '.pdf')

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
