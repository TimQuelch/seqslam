import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

argparser = argparse.ArgumentParser(description="Process PR curves")

argparser.add_argument('-s', '--show', action='store_true', help='Display figures as windows')
argparser.add_argument('-w', '--write', action='store_true', help='Write figures to files')

datafile = 'pr.json'
fileprefix = 'pr'

def main(args):
    with open(datafile) as b:
        raw = json.load(b)

    figs = []

    d = pd.DataFrame(raw["data"])
    print(d)

    fig, ax = plt.subplots()
    ax = d.plot(x="Recall", y="Precision", ax=ax)
    figs.append((fig, 'pr'))

    if args.show:
        plt.show()

    if args.write:
        for fig, name in figs:
            fig.savefig(fileprefix + '-' + name + '.pdf')

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
