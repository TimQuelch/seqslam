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

slName = '$l_s$'
ntName = '$n_{traj}$'
wsName = '$w_s$'
f1Name = '$F_1$'
timeName = 'Time ($\mathrm{ms^{-1}}$)'

originalExt = '.json.gz'
processedExt = '.csv'
sweepPrefix = 'sweep'
timesweepPrefix = 'timesweep'
sweeps1d = [('sl', [slName]),
            ('ws', [wsName]),
            ('nt', [ntName])]
sweeps2d = [('sl-nt', [slName, ntName]),
            ('sl-ws', [slName, wsName]),
            ('ws-nt', [wsName, ntName])]
timesweeps = [('sl', [slName, 'Max time']),
              ('ws', [wsName, 'Max time']),
              ('nt', [ntName, 'Max time'])]
prs = [('sl', slName, [2, 5, 10, 20, 40]),
       ('ws', wsName, [2, 5, 10, 20, 40]),
       ('nt', ntName, [2, 5, 10, 20, 40])]

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

    d = d.rename(columns={'F1 Score': f1Name,
                          'Sequence length': slName,
                          'Patch window size': wsName,
                          'Number of trajectories': ntName})

    nu = d.nunique(axis=0)
    old = d
    d = d.drop(nu[nu == 1].index, axis=1)
    d['Iterations'] = old['Iterations']
    d['Difference matrix calculation'] = d['Difference matrix calculation'] / d['Iterations']
    d['Difference matrix enhancement'] = d['Difference matrix enhancement'] / d['Iterations']
    d['Sequence search'] = d['Sequence search'] / d['Iterations']
    d[timeName] = (d['Difference matrix calculation'] +
                 d['Sequence search'] +
                 d['Difference matrix enhancement'])

    full = d
    idx = d.groupby(variables)[f1Name].transform(max) == d[f1Name]
    d = d[idx]
    idx = d.groupby(variables)[timeName].transform(min) == d[timeName]
    d = d[idx]
    d = d.drop_duplicates(subset=variables + [f1Name, timeName])
    return full, d

def main(args):
    figs = []

    for sweep, variables in sweeps1d:
        full, d = loadSweep(sweepPrefix, sweep, variables)
        v = variables[0]

        fig, ax = plt.subplots()
        d.set_index(v)[[f1Name, timeName]].plot(style='-o', secondary_y=[timeName], ax=ax)
        axes = fig.get_axes()
        axes[0].set_ylabel(f1Name)
        axes[0].set_ylim([0, 1])
        axes[1].set_ylabel(timeName)
        axes[1].set_ylim([310, 360])
        axes[0].get_legend().set_bbox_to_anchor((1, 0.3))
        figs.append((fig, sweepPrefix + '-' + sweep))

    for sweep, variables in timesweeps:
        full, d = loadSweep(timesweepPrefix, sweep, variables)

        maxTimeName = 'Max time ($\mathrm{ms^{-1}}$)'
        d[maxTimeName] = d['Max time']
        d = d.set_index(maxTimeName)

        fig, ax = plt.subplots()
        d[[f1Name, timeName, 'Max time']].plot(style=['-o', '-o', ':'], secondary_y=[timeName, 'Max time'], ax=ax)
        axes = fig.get_axes()
        axes[0].set_ylabel(f1Name)
        axes[0].set_ylim([0, 1])
        axes[1].set_ylabel(timeName)
        axes[1].set_ylim([310, 360])
        axes[0].get_legend().set_bbox_to_anchor((1, 0.3))
        figs.append((fig, timesweepPrefix + '-' + sweep))

    for sweep, variables in sweeps2d:
        full, d = loadSweep(sweepPrefix, sweep, variables)

        timeGrid = d[[variables[0], variables[1], timeName]]
        timeGrid = timeGrid.set_index([variables[1], variables[0]]).sort_index()
        timeGrid = timeGrid.unstack(level=variables[0])
        timeGrid = ndimage.gaussian_filter(timeGrid.to_numpy(), args.gaussian) if args.gaussian > 0 else timeGrid.to_numpy()

        f1Grid = d[[variables[0], variables[1], f1Name]]
        f1Grid = f1Grid.set_index([variables[1], variables[0]]).sort_index()
        f1Grid = f1Grid.unstack(level=variables[0])
        f1Grid = ndimage.gaussian_filter(f1Grid.to_numpy(), args.gaussian) if args.gaussian > 0 else f1Grid.to_numpy()

        X, Y = np.meshgrid(d[variables[0]].unique(), d[variables[1]].unique())

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, timeGrid)
        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        ax.set_zlabel(timeName)
        ax.view_init(elev=25, azim=-135)
        figs.append((fig, sweepPrefix + '-' + sweep + '-time-surf'))

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, f1Grid)
        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        ax.set_zlabel(f1Name)
        ax.view_init(elev=25, azim=-135)
        figs.append((fig, sweepPrefix + '-' + sweep + '-f1-surf'))

        fig, ax = plt.subplots()
        ax.contourf(X, Y, timeGrid)
        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        figs.append((fig, sweepPrefix + '-' + sweep + '-time-contour'))

        fig, ax = plt.subplots()
        ax.contourf(X, Y, f1Grid)
        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        figs.append((fig, sweepPrefix + '-' + sweep + '-f1-contour'))

    for sweep, variable, values in prs:
        full, d = loadSweep(sweepPrefix, sweep, [variable])
        pr = full.set_index(variable).loc[values].reset_index()
        pr = pr.set_index('Recall')

        fig, ax = plt.subplots()
        pr.groupby(variable)['Precision'].plot(style='-', legend=True, ax=ax)
        ax.set_prop_cycle(None)
        pr.groupby(variable)[f1Name].plot(style=':', legend=False, ax=ax)
        ax.set_ylabel('Precision')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.get_legend().set_title(variable)
        figs.append((fig, 'pr-' + sweep))

    if args.show:
        plt.show()

    if args.write:
        for fig, name in figs:
            fig.savefig(name + '.pdf')

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
