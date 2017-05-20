"""
Mihail Dunaev
May 2017

Common functions used in all tests.
"""

import numpy as np

def read_data(fname):
    with open(fname, "r") as f:
        lines = f.readlines()
        N = len(lines)
        num_features = len(lines[0].split()) - 1
        xs = np.zeros((N, num_features))
        ys = np.zeros(N)
        for line, i in zip(lines, xrange(N)):
            vals = line.split()
            xs[i] = map(float, vals[0:-1])
            ys[i] = int(vals[-1])
    return (xs, ys)
