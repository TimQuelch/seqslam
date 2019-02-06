import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

argparser = argparse.ArgumentParser(description="Process benchmark results")

argparser.add_argument('-s', '--show', action='store_true', help='Display figures as windows')
argparser.add_argument('-w', '--write', action='store_true', help='Write figures to files')

throughputLabel = 'Throughput ($\mathrm{GiBs^{-1}}$)'
itemRateLabel = 'Million Items per Second'

def setYAxis(ax, label, top=None):
    ax.set_ylabel(label)
    ax.set_ylim(bottom=0, top=top)
    return ax

def main(args):
    with open('benchmarks-enhance.json') as b:
        raw = json.load(b)

    figs = []

    d = pd.DataFrame(raw['benchmarks'])

    splitNames = d['name'].str.split('/', expand=True)
    d['Method'] = splitNames[0]
    d['Window Size'] = pd.to_numeric(splitNames[1], downcast='integer')
    d['n Pixels per Thread'] = pd.to_numeric(splitNames[2], downcast='integer')

    wsizeName = '$w_{size}$'
    nloadName = '$n_{load}$'
    d = d.rename(columns=lambda s: s.replace('_', ' '))
    d = d.rename(columns=lambda s: s.title())
    d = d.rename(columns=lambda s: s.replace('Per', 'per'))
    d = d.rename(columns=lambda s: s.replace('Cpu', 'CPU'))
    d = d.rename(columns={'Window Size': wsizeName, 'N Pixels per Thread': nloadName})

    d['Method'] = d['Method'].str.replace('Enhancement', '')
    d['Method'] = d['Method'].replace(to_replace={'cpu': 'CPU'})
    d['Method'] = d['Method'].replace(to_replace={'gpu': 'GPU'})

    d['GiB per Second'] = d['Bytes per Second'] / 2**(10*3)
    d['Million Items per Second'] = d['Items per Second'] / 1e6
    d = d.drop(columns=['Time Unit', 'Name'])

    d = d.set_index(['Method', wsizeName, nloadName])
    d = d.sort_index()

    dr = d[['GiB per Second', 'Million Items per Second']]

    cpu = dr.loc[('CPU')]
    cpu = cpu.reset_index(level=[nloadName], drop=True)

    fig, ax = plt.subplots()
    ax = cpu['GiB per Second'].plot(style='o-', ax=ax)
    ax = setYAxis(ax, throughputLabel)
    figs.append((fig, 'cpu-rate'))

    fig, ax = plt.subplots()
    ax = cpu['Million Items per Second'].plot(style='o-', ax=ax)
    ax = setYAxis(ax, itemRateLabel)
    figs.append((fig, 'cpu-items'))

    #gpu = dr.loc['GPU']
    #gpu = gpu.unstack(level=wsizeName)

    #fig, ax = plt.subplots()
    #ax = gpu['GiB per Second'].plot(style='o-', ax=ax)
    #ax = setYAxis(ax, throughputLabel)
    #figs.append((fig, 'gpu-rate-1'))

    #fig, ax = plt.subplots()
    #ax = gpu['Million Items per Second'].plot(style='o-', ax=ax)
    #ax = setYAxis(ax, itemRateLabel)
    #figs.append((fig, 'gpu-items-1'))

    gpu = dr.loc['GPU']
    gpu = gpu.unstack(level=nloadName)

    fig, ax = plt.subplots()
    ax = gpu['GiB per Second'].plot(style='o-', ax=ax)
    ax = setYAxis(ax, throughputLabel)
    figs.append((fig, 'gpu-rate-2'))

    fig, ax = plt.subplots()
    ax = gpu['Million Items per Second'].plot(style='o-', ax=ax)
    ax = setYAxis(ax, itemRateLabel)
    figs.append((fig, 'gpu-items-2'))

    if args.show:
        plt.show()

    if args.write:
        for fig, name in figs:
            fig.savefig(name + '.pdf')

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
