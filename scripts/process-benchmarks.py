import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

argparser = argparse.ArgumentParser(description="Process benchmark results")

argparser.add_argument('-s', '--show', action='store_true', help='Display figures as windows')
argparser.add_argument('-w', '--write', action='store_true', help='Write figures to files')

def setYAxis(ax, top=None):
    ax.set_ylabel('Throughput ($\mathrm{GiBs^{-1}}$)')
    ax.set_ylim(bottom=0, top=top)
    return ax

def main(args):
    with open('benchmarks.json') as b:
        raw = json.load(b)

    figs = []

    d = pd.DataFrame(raw['benchmarks'])

    splitNames = d['name'].str.split('/', expand=True)
    splitLabel = splitNames[1].str.split('_', expand=True)
    splitLabel[0] = splitLabel[0].replace(to_replace={'small': 'Small', 'large': 'Large'})
    splitLabel[1] = splitLabel[1].replace(to_replace={None: '-'})
    d['Method'] = splitNames[0]
    d['Tile size'] = pd.to_numeric(splitNames[2], downcast='integer')
    d['n Pixels per Thread'] = pd.to_numeric(splitNames[3], downcast='integer')
    d['Dataset'] = splitLabel[0]
    d['Label'] = splitLabel[1]

    tsizeName = '$t_{size}$'
    nloadName = '$n_{load}$'
    d = d.rename(columns=lambda s: s.replace('_', ' '))
    d = d.rename(columns=lambda s: s.title())
    d = d.rename(columns=lambda s: s.replace('Per', 'per'))
    d = d.rename(columns=lambda s: s.replace('Cpu', 'CPU'))
    d = d.rename(columns={'Tile Size': tsizeName, 'N Pixels per Thread': nloadName})

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

    d = d.set_index(['Method', 'Dataset', 'Label', tsizeName, nloadName])
    d = d.sort_index()

    dr = d['GiB per Second']

    cpu = dr.loc[('CPU')]
    cpu = cpu.reset_index(level=['Label', nloadName], drop=True)
    cpu = cpu.unstack(level='Dataset')
    ax = cpu.plot(style='o-')
    ax = setYAxis(ax)
    figs.append((ax.get_figure(), 'cpu'))

    gpu = dr.loc[('GPU')]
    gpu = gpu.unstack(level=nloadName)
    ax1 = gpu.loc[('Large', 'Best')].plot(style='o-')
    ax2 = gpu.loc[('Small', 'Best')].plot(style='o-')
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1 = setYAxis(ax1, top=ymax)
    ax2 = setYAxis(ax2, top=ymax)
    figs.append((ax1.get_figure(), 'gpu-large'))
    figs.append((ax2.get_figure(), 'gpu-small'))

    early = dr.loc[('GPU', 'Small',
                    ['Naive', 'Parallel save', 'Continuous index', 'Two diffs', 'Warp reduce'])]
    early = early.reset_index(level=['Method', 'Dataset', nloadName], drop=True)
    early = early.unstack(level='Label')
    ax = early.plot(style='o-')
    ax = setYAxis(ax)
    figs.append((ax.get_figure(), 'gpu-early'))

    copies = dr.loc[(['GPU', 'GPU (with copy and context)', 'GPU (with copy)'],
                     'Small',
                     slice(None),
                     slice(None),
                     8)]
    copies = copies.reset_index(level=['Dataset', 'Label', nloadName], drop=True)
    copies = copies.unstack(level='Method')
    ax = copies.plot(style='o-')
    ax = setYAxis(ax)
    figs.append((ax.get_figure(), 'gpu-copies-and-context'))

    maxes = dr[dr.groupby(level=['Method', 'Dataset', 'Label']).idxmax()]
    print('Small dataset')
    print(maxes.loc[(slice(None), 'Small')])
    print('Large dataset')
    print(maxes.loc[(slice(None), 'Large')])

    if args.show:
        plt.show()

    if args.write:
        for fig, name in figs:
            fig.savefig(name + '.pdf')

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
