import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json
import argparse
import os.path
import gzip
from scipy import ndimage

argparser = argparse.ArgumentParser(description="Process PR curves")

argparser.add_argument('-s', '--show', action='store_true', help='Display figures as windows')
argparser.add_argument('-w', '--write', action='store_true', help='Write figures to files')
argparser.add_argument('-u', '--update', action='store_true', help='Force update csv file')
argparser.add_argument('-g', '--gaussian', type=int, default=-1, help='Size of gaussian kernel to smooth surfaces')

originalExt = '.json.gz'
processedExt = '.csv'
sweepPrefix = 'sweep'
timesweepPrefix = 'timesweep'
sweeps1d = [('sl', ['Sequence length']),
            ('ws', ['Patch window size']),
            ('nt', ['Number of trajectories'])]
sweeps2d = [('sl-nt', ['Sequence length', 'Number of trajectories']),
            ('sl-ws', ['Sequence length', 'Patch window size']),
            ('ws-nt', ['Patch window size', 'Number of trajectories'])]
timesweeps = [('sl', ['Sequence length', 'Max time']),
              ('ws', ['Patch window size', 'Max time']),
              ('nt', ['Number of trajectories', 'Max time'])]

def checkUpdateRequired(original, processed):
    if not os.path.isfile(processed):
        return True

    oTime = os.path.getmtime(original)
    pTime = os.path.getmtime(processed)

    if oTime > pTime:
        return True

    return False

def loadSweep(prefix, sweep, variables):
    datafileJson = prefix + '-' + sweep + originalExt
    datafileCsv = prefix + '-' + sweep + processedExt
    if checkUpdateRequired(datafileJson, datafileCsv) or args.update:
        with gzip.open(datafileJson) as b:
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
    old = d
    d = d.drop(nu[nu == 1].index, axis=1)
    d['Iterations'] = old['Iterations']
    d['Difference matrix calculation'] = d['Difference matrix calculation'] / d['Iterations']
    d['Difference matrix enhancement'] = d['Difference matrix enhancement'] / d['Iterations']
    d['Sequence search'] = d['Sequence search'] / d['Iterations']
    d['Time'] = (d['Difference matrix calculation'] +
                 d['Sequence search'] +
                 d['Difference matrix enhancement'])

    full = d
    idx = d.groupby(variables)['F1 Score'].transform(max) == d['F1 Score']
    d = d[idx]
    idx = d.groupby(variables)['Time'].transform(min) == d['Time']
    d = d[idx]
    d = d.drop_duplicates(subset=variables + ['F1 Score', 'Time'])
    return full, d

def main(args):
    figs = []

    for sweep, variables in sweeps1d:
        full, d = loadSweep(sweepPrefix, sweep, variables)
        v = variables[0]

        fig, ax = plt.subplots()
        d.set_index(v)[['F1 Score', 'Time']].plot(style='-o', secondary_y=['Time'], ax=ax)
        axes = fig.get_axes()
        axes[0].set_ylabel('F1 Score')
        axes[0].set_ylim([0, 1])
        axes[1].set_ylabel('Time')
        axes[1].set_ylim([300, 360])

    for sweep, variables in timesweeps:
        full, d = loadSweep(timesweepPrefix, sweep, variables)

        d['Max time 2'] = d['Max time']
        d = d.set_index('Max time 2')

        fig, ax = plt.subplots()
        d[['F1 Score', 'Time', 'Max time']].plot(style=['-o', '-o', ':'], secondary_y=['Time', 'Max time'], ax=ax)
        axes = fig.get_axes()
        axes[0].set_xlabel('Max time')
        axes[0].set_ylabel('F1 Score')
        axes[0].set_ylim([0, 1])
        axes[1].set_ylabel('Time')
        axes[1].set_ylim([300, 360])

    for sweep, variables in sweeps2d:
        full, d = loadSweep(sweepPrefix, sweep, variables)

        timeGrid = d[[variables[0], variables[1], 'Time']]
        timeGrid = timeGrid.set_index([variables[1], variables[0]]).sort_index()
        timeGrid = timeGrid.unstack(level=variables[0])
        timeGrid = ndimage.gaussian_filter(timeGrid.to_numpy(), args.gaussian) if args.gaussian > 0 else timeGrid.to_numpy()

        f1Grid = d[[variables[0], variables[1], 'F1 Score']]
        f1Grid = f1Grid.set_index([variables[1], variables[0]]).sort_index()
        f1Grid = f1Grid.unstack(level=variables[0])
        f1Grid = ndimage.gaussian_filter(f1Grid.to_numpy(), args.gaussian) if args.gaussian > 0 else f1Grid.to_numpy()

        X, Y = np.meshgrid(d[variables[0]].unique(), d[variables[1]].unique())

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, timeGrid)
        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        ax.set_zlabel('Time')
        figs.append((fig, sweep + '-time-surf'))

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, f1Grid)
        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        ax.set_zlabel('F1')
        figs.append((fig, sweep + '-f1-surf'))

        fig, ax = plt.subplots()
        ax.contourf(X, Y, timeGrid)
        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        figs.append((fig, sweep + '-time-contour'))

        fig, ax = plt.subplots()
        ax.contourf(X, Y, f1Grid)
        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        figs.append((fig, sweep + '-f1-contour'))

    if args.show:
        plt.show()

    if args.write:
        for fig, name in figs:
            fig.savefig(prefix + '-' + name + '.pdf')

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
