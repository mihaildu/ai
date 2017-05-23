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

def plot_all(xs, ys, nn, plot3d=True, colormesh=False):
    """
    Generic plotting.
      plot3d & colormesh can't be True
      at the same time (easy fix).
    """
    # switch backend - TODO fix this
    plt.switch_backend("TKagg")
    if plot3d:
        fig = plt.figure(1)
        ax = fig.gca(projection="3d")

    N = len(xs)
    xs0 = np.array([xs[i] for i in xrange(N) if ys[i] == 0])
    xs1 = np.array([xs[i] for i in xrange(N) if ys[i] == 1])

    # plot initial data
    plt.plot(xs0[:,0], xs0[:,1], "bo")
    plt.plot(xs1[:,0], xs1[:,1], "rx")

    # set range for x/y axis
    xmin = xs[:,0].min()
    xmax = xs[:,0].max()
    ymin = xs[:,1].min()
    ymax = xs[:,1].max()
    plt.xlim([xmin - 1, xmax + 1])
    plt.ylim([ymin - 1, ymax + 1])

    # plot best fit (3D)
    if plot3d or colormesh:
        num_pts = 100
        (wxs, wys) = np.meshgrid(np.linspace(xmin - 1, xmax + 1, num_pts),
                                 np.linspace(ymin - 1, ymax + 1, num_pts))
        if plot3d:
            wzs = nn.predict_proba(np.concatenate((wxs.reshape(wxs.size, 1),
                               wys.reshape(wys.size, 1)), axis=1))[:,1]
            ax.plot_surface(wxs, wys, wzs.reshape(wxs.shape))
        if colormesh:
            my_cmap = colors.ListedColormap(["b", "r"])
            wzs = nn.predict(np.concatenate((wxs.reshape(wxs.size, 1),
                                         wys.reshape(wys.size, 1)), axis=1))
            plt.pcolormesh(wxs, wys, wzs.reshape(wxs.shape),
                           cmap=my_cmap, alpha=0.5)
    plt.show()
