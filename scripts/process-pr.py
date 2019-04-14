import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

argparser = argparse.ArgumentParser(description="Process PR curves")

argparser.add_argument('-s', '--show', action='store_true', help='Display figures as windows')
argparser.add_argument('-w', '--write', action='store_true', help='Write figures to files')

vals = [5, 10, 15, 20, 25, 30]
datafiles = [('window', vals), ('seqlength', vals), ('ntraj', vals)]
prefix = 'pr'
ext = '.json'

def main(args):
    figs = []

    for parameter, values in datafiles:
        ds = []
        for val in values:
            with open(prefix + '-' + parameter + '-' + str(val) + '.json') as b:
                raw = json.load(b)

            ds.append(pd.DataFrame(raw["data"]).set_index('Recall'))

        d = pd.concat(ds, keys=values, names=[parameter])
        d = d.reset_index()
        d = d.set_index('Recall')

        fig, ax = plt.subplots()
        d.groupby(parameter)['Precision'].plot(style='-', ax=ax, legend=True)
        ax.set_ylabel('Precision')
        figs.append((fig, prefix + '-' + parameter))

    if args.show:
        plt.show()

    if args.write:
        for fig, name in figs:
            fig.savefig(prefix + '-' + name + '.pdf')

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
