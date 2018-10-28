import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

def main():
    with open('benchmarks.json') as b:
        raw = json.load(b)

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
    d['GiB per Second'] = d['Bytes per Second'] / 2**(10*3)
    d = d.drop(columns=['Time Unit', 'Name'])

    d = d.set_index(['Method', 'Dataset', 'Label', 'Tile Size', 'N Pixels per Thread'])
    d = d.sort_index()

    dr = d['GiB per Second']

    cpu = dr.loc[('CPU')]
    cpu = cpu.reset_index(level=['Label', 'N Pixels per Thread'], drop=True)
    cpu = cpu.unstack(level='Dataset')
    cpu.plot(style='o-')

    gpu = dr.loc[('GPU')]
    gpu = gpu.unstack(level='N Pixels per Thread')
    gpu.loc[('Large', 'best')].plot(style='o-')
    gpu.loc[('Small', 'best')].plot(style='o-')

    early = dr.loc[('GPU', 'Small', ['naive', 'parallelsave', 'continuousindex', 'twodiffs', 'warpreduce'])]
    early = early.reset_index(level=['Method', 'Dataset', 'N Pixels per Thread'], drop=True)
    early = early.unstack(level='Label')
    early.plot(style='o-')

    plt.show()

if __name__ == '__main__':
    main()
