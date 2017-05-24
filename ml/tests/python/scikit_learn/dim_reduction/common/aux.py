"""
Mihail Dunaev
May 2017

Common functions used in some tests.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

def read_data(fname, output_layers=1):
    """ Generic data read from file """
    with open(fname, "r") as f:
        lines = f.readlines()
        N = len(lines)
        num_features = len(lines[0].split()) - output_layers
        xs = np.zeros((N, num_features))
        if output_layers == 1:
            ys = np.zeros(N)
        else:
            ys = np.zeros((N, output_layers))
        for line, i in zip(lines, xrange(N)):
            vals = line.split()
            xs[i] = map(float, vals[0:-output_layers])
            if output_layers == 1:
                ys[i] = int(vals[-1])
            else:
                ys[i] = map(int, vals[-output_layers:])
    return (xs, ys)
