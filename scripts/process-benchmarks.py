import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

argparser = argparse.ArgumentParser(description="Process benchmark results")

argparser.add_argument('-s', '--show', action='store_true', help='Display figures as windows')
argparser.add_argument('-w', '--write', action='store_true', help='Write figures to files')

def setYAxis(ax):
    ax.set_ylabel('Data Throughput (GiB/s)')
    return ax

def main(args):
    with open('benchmarks.json') as b:
        raw = json.load(b)

    figs = []

    d = pd.DataFrame(raw['benchmarks'])

    splitNames = d['name'].str.split('/', expand=True)
    splitLabel = splitNames[1].str.split('_', expand=True)
    splitLabel[0] = splitLabel[0].replace(to_replace={'small': 'Small', 'large': 'Large'})
    d['Method'] = splitNames[0]
    d['Tile size'] = pd.to_numeric(splitNames[2], downcast='integer')
    d['n Pixels per Thread'] = pd.to_numeric(splitNames[3], downcast='integer')
    d['Dataset'] = splitLabel[0]
    d['Label'] = splitLabel[1]

    d = d.rename(columns=lambda s: s.replace('_', ' '))
    d = d.rename(columns=lambda s: s.title())
    d = d.rename(columns=lambda s: s.replace('Per', 'per'))
    d = d.rename(columns=lambda s: s.replace('Cpu', 'CPU'))

    d['Method'] = d['Method'].replace(to_replace={'DifferenceMatrix': ''}, regex=True)
    d['Method'] = d['Method'].replace(to_replace={'cpu': 'CPU',
                                                  'gpu': 'GPU',
                                                  'gpuWithCopyAndContext':
                                                  'GPU (with copy and context)',
                                                  'gpuWithCopy': 'GPU (with copy)'})
    d['Label'] = d['Label'].replace(to_replace={'best': 'Best',
                                                'continuousindex': 'Continuous index',
                                                'naive': 'Naive',
                                                'parallelsave': 'Parallel save',
                                                'twodiffs': 'Two diffs',
                                                'warpreduce': 'Warp reduce'})
    d['GiB per Second'] = d['Bytes per Second'] / 2**(10*3)
    d = d.drop(columns=['Time Unit', 'Name'])

    d = d.set_index(['Method', 'Dataset', 'Label', 'Tile Size', 'N Pixels per Thread'])
    d = d.sort_index()

    dr = d['GiB per Second']

    cpu = dr.loc[('CPU')]
    cpu = cpu.reset_index(level=['Label', 'N Pixels per Thread'], drop=True)
    cpu = cpu.unstack(level='Dataset')
    ax = cpu.plot(style='o-')
    ax = setYAxis(ax)
    figs.append((ax.get_figure(), 'cpu'))

    gpu = dr.loc[('GPU')]
    gpu = gpu.unstack(level='N Pixels per Thread')
    ax = gpu.loc[('Large', 'Best')].plot(style='o-')
    ax = setYAxis(ax)
    figs.append((ax.get_figure(), 'gpu-large'))
    ax = gpu.loc[('Small', 'Best')].plot(style='o-')
    ax = setYAxis(ax)
    figs.append((ax.get_figure(), 'gpu-small'))

    early = dr.loc[('GPU', 'Small',
                    ['Naive', 'Parallel save', 'Continuous index', 'Two diffs', 'Warp reduce'])]
    early = early.reset_index(level=['Method', 'Dataset', 'N Pixels per Thread'], drop=True)
    early = early.unstack(level='Label')
    ax = early.plot(style='o-')
    ax = setYAxis(ax)
    figs.append((ax.get_figure(), 'gpu-early'))

    if args.show:
        plt.show()

    if args.write:
        for fig, name in figs:
            fig.savefig(name + '.pdf')

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
