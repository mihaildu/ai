"""
Mihail Dunaev
May 2017

Simple test that finds 2 clusters in 2D data using kmeans and
  dbscan. Data is in 'kmeans.in'.

For kmeans & dbscan I used scikit-learn:
  sklearn.cluster.KMeans
  sklearn.cluster.DBSCAN
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

def main():
    """ wrapper for test1 """
    test1(use_dbscan=False)

def test1(use_dbscan=False):
    (xs, N) = read_data("../../../input/kmeans.in")

    if use_dbscan:
        ds = DBSCAN(eps=1.2, min_samples=2)
        ds.fit(xs)
        (ks, cs) = (None, ds.labels_)
    else:
        km = KMeans(n_clusters=2, n_init=10)
        km.fit(xs)
        (ks, cs) = (km.cluster_centers_, km.labels_)

    plot_all(xs, cs, N, ks)

def plot_all(xs, cs, N, ks=None):
    # switch backend - TODO fix this
    plt.switch_backend("TKagg")

    # plot labeled points
    xs0 = np.array([xs[i] for i in xrange(N) if cs[i] == 0])
    xs1 = np.array([xs[i] for i in xrange(N) if cs[i] == 1])
    xs2 = np.array([xs[i] for i in xrange(N) if cs[i] == -1])
    if len(xs0) > 0:
        plt.plot(xs0[:,0], xs0[:,1], "ro")
    if len(xs1) > 0:
        plt.plot(xs1[:,0], xs1[:,1], "bo")
    if len(xs2) > 0:
        plt.plot(xs2[:,0], xs2[:,1], "ko")

    # set range for x/y axis
    xmin = xs[:,0].min()
    xmax = xs[:,0].max()
    ymin = xs[:,1].min()
    ymax = xs[:,1].max()
    plt.xlim([xmin - 1, xmax + 1])
    plt.ylim([ymin - 1, ymax + 1])

    # plot centroids
    if ks is not None:
        plt.plot(ks[0][0], ks[0][1], "rx")
        plt.plot(ks[1][0], ks[1][1], "bx")

    plt.show()

def read_data(fname):
    """ Generic data read from file """
    with open(fname, "r") as f:
        lines = f.readlines()
        N = len(lines)
        num_features = len(lines[0].split())
        xs = np.zeros((N, num_features))
        for line, i in zip(lines, xrange(N)):
            vals = line.split()
            xs[i] = map(float, vals)
    return (xs, N)

main()
