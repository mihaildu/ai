"""
Mihail Dunaev
May 2017

Simple test that performs k-nearest neighbors (knn) on 2D data
  from 'logistic_multiclass.in'.

For knn I used scikit-learn:
  sklearn.neighbors.KNearestNeighbors
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def test1():
    (xs, ys) = read_data("../../../input/logistic_multiclass.in")

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(xs, ys)

    xpred = np.array([6, 2.1, 4.5, 4.5, 6, 7, 6, 6.1, 3,
                      6, 2, 1.5, 5, 4]).reshape(7, 2);
    ypred = knn.predict(xpred)
    plot_all(xs, ys, xpred, ypred)

def plot_all(xs, ys, xpred, ypred):
    # switch backend - TODO fix this
    plt.switch_backend("TKagg")

    # plot labeled points
    N = len(xs)
    xs0 = np.array([xs[i] for i in xrange(N) if ys[i] == 0])
    xs1 = np.array([xs[i] for i in xrange(N) if ys[i] == 1])
    xs2 = np.array([xs[i] for i in xrange(N) if ys[i] == 2])
    plt.plot(xs0[:,0], xs0[:,1], "bo")
    plt.plot(xs1[:,0], xs1[:,1], "rd")
    plt.plot(xs2[:,0], xs2[:,1], "gs")

    # set range for x/y axis
    xmin = xs[:,0].min()
    xmax = xs[:,0].max()
    ymin = xs[:,1].min()
    ymax = xs[:,1].max()
    plt.xlim([xmin - 1, xmax + 1])
    plt.ylim([ymin - 1, ymax + 1])

    # plot predicted points
    npred = xpred.shape[0]
    for i in xrange(npred):
        if ypred[i] == 0:
            plt.plot(xpred[i][0], xpred[i][1], "bx")
        elif ypred[i] == 1:
            plt.plot(xpred[i][0], xpred[i][1], "rx")
        elif ypred[i] == 2:
            plt.plot(xpred[i][0], xpred[i][1], "gx")
        else:
            plt.plot(xpred[i][0], xpred[i][1], "kx")

    plt.show()

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

test1()
