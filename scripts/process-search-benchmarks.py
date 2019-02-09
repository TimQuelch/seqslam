import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

argparser = argparse.ArgumentParser(description="Process sequence search benchmark results")

argparser.add_argument('-s', '--show', action='store_true', help='Display figures as windows')
argparser.add_argument('-w', '--write', action='store_true', help='Write figures to files')

datafile = 'benchmarks-search.json'
fileprefix = 'search'

throughputLabel = 'Throughput ($\mathrm{GiBs^{-1}}$)'
itemRateLabel = 'Million Items per Second'

def setYAxis(ax, label, top=None):
    ax.set_ylabel(label)
    ax.set_ylim(bottom=0, top=top)
    return ax

def main(args):
    with open(datafile) as b:
        raw = json.load(b)

    figs = []

    vRangeSize = 'Large'

    d = pd.DataFrame(raw['benchmarks'])
    d = d.loc[~d['name'].str.contains('IndexOffsets')]

    splitNames = d['name'].str.split('/', expand=True)
    d['Method'] = splitNames[0]
    d['Velocity Range'] = splitNames[1].str.title()
    d['Sequence Length'] = pd.to_numeric(splitNames[2], downcast='integer')
    d['n Trajectories'] = pd.to_numeric(splitNames[3], downcast='integer')
    d['n Pixels per Thread'] = pd.to_numeric(splitNames[4], downcast='integer')

    seqLengthName = '$l_{s}$'
    ntrajName = '$n_{traj}$'
    nloadName = '$n_{load}$'
    d = d.rename(columns=lambda s: s.replace('_', ' '))
    d = d.rename(columns=lambda s: s.title())
    d = d.rename(columns=lambda s: s.replace('Per', 'per'))
    d = d.rename(columns=lambda s: s.replace('Cpu', 'CPU'))
    d = d.rename(columns={'Sequence Length': seqLengthName,
                          'N Trajectories': ntrajName,
                          'N Pixels per Thread': nloadName})

    d['Method'] = d['Method'].str.replace('Search', '')
    d['Method'] = d['Method'].replace(to_replace={'cpu': 'CPU'})
    d['Method'] = d['Method'].replace(to_replace={'gpu': 'GPU'})

    d['GiB per Second'] = d['Bytes per Second'] / 2**(10*3)
    d['Million Items per Second'] = d['Items per Second'] / 1e6
    d = d.drop(columns=['Time Unit', 'Name'])

    d = d.set_index(['Method', 'Velocity Range', seqLengthName, ntrajName, nloadName])
    d = d.sort_index()

    dr = d[['GiB per Second', 'Million Items per Second']]

    cpu = dr.loc[('CPU', vRangeSize)]
    cpu = cpu.reset_index(level=[nloadName], drop=True)
    cpu = cpu.unstack(level=[ntrajName])

    fig, ax = plt.subplots()
    ax = cpu['GiB per Second'].plot(style='o-', ax=ax)
    ax = setYAxis(ax, throughputLabel)
    figs.append((fig, 'cpu-rate'))

    fig, ax = plt.subplots()
    ax = cpu['Million Items per Second'].plot(style='o-', ax=ax)
    ax = setYAxis(ax, itemRateLabel)
    figs.append((fig, 'cpu-items'))

    gpu = dr.loc[('GPU', vRangeSize, slice(None), slice(None), 6)]
    gpu = gpu.reset_index(level=['Method', 'Velocity Range', nloadName], drop=True)
    gpu = gpu.unstack(level=[ntrajName])

    fig, ax = plt.subplots()
    ax = gpu['GiB per Second'].plot(style='o-', ax=ax)
    ax = setYAxis(ax, throughputLabel)
    figs.append((fig, 'gpu-rate'))

    fig, ax = plt.subplots()
    ax = gpu['Million Items per Second'].plot(style='o-', ax=ax)
    ax = setYAxis(ax, itemRateLabel)
    figs.append((fig, 'gpu-items'))

    speedup = dr.loc['GPU', vRangeSize] / dr.loc['CPU', vRangeSize].reset_index(level=[nloadName], drop=True)
    speedup = speedup.loc(axis=0)[(slice(None), slice(None), 6)].reset_index(level=[nloadName], drop=True)
    speedup = speedup.unstack(level=ntrajName)

    fig, ax = plt.subplots()
    ax = speedup['GiB per Second'].plot(style='o-', ax=ax)
    ax = setYAxis(ax, 'Speedup relative to CPU')
    figs.append((fig, 'speedup'))

    if args.show:
        plt.show()

    if args.write:
        for fig, name in figs:
            fig.savefig(fileprefix + '-' + name + '.pdf')

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
