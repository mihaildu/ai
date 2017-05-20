"""
Mihail Dunaev
May 2017

Simple test that tries to separate 3 sets of 2D data on a plane
  using one-vs-all linear logistic regression. Data is in 
  'logistic_multiclass.in'.

For logistic regression I used scikit-learn:
  sklearn.linear_model.LogisticRegression
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from scipy.special import expit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

from common.aux import read_data

def test5_multiclass():
    (xs, ys) = read_data("../../../input/logistic_multiclass.in")

    # train logistic regression model with sklearn
    lr = linear_model.LogisticRegression(C=1e5)
    lr.fit(xs, ys)

    # predict some values
    xpred = np.array([6, 2.1, 4.5, 4.5, 6, 7, 6, 6.1, 3, 6, 2, 1.5, 5, 4]).reshape(7, 2)
    ypred = lr.predict(xpred)
    print(ypred.tolist())

    plot_all(xs, ys, lr, xpred, ypred, show_lines=True, show_colormesh=False)

def plot_all(xs, ys, lr, xpred, ypred, show_lines=True, show_colormesh=False):
    # switch backend - TODO fix this
    plt.switch_backend("TKagg")

    N = len(xs)
    xs0 = np.array([xs[i] for i in xrange(N) if ys[i] == 0])
    xs1 = np.array([xs[i] for i in xrange(N) if ys[i] == 1])
    xs2 = np.array([xs[i] for i in xrange(N) if ys[i] == 2])

    # plot initial data
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

    # plot separating lines
    # lr.coef_/intercept_ looks ugly
    if show_lines:
        W = np.zeros((3, 3))
        W[:,0:2] = lr.coef_[:,0:2]
        W[:,2] = lr.intercept_
        x0 = np.array([[xmin], [xmax]])
        y0 = np.array([[-(W[0][2] + W[0][0] * xmin) / W[0][1]],
                       [-(W[0][2] + W[0][0] * xmax) / W[0][1]]])
        y1 = np.array([[-(W[1][2] + W[1][0] * xmin) / W[1][1]],
                       [-(W[1][2] + W[1][0] * xmax) / W[1][1]]])
        y2 = np.array([[-(W[2][2] + W[2][0] * xmin) / W[2][1]],
                       [-(W[2][2] + W[2][0] * xmax) / W[2][1]]])
        plt.plot(x0, y0, "b-")
        plt.plot(x0, y1, "r-")
        plt.plot(x0, y2, "g-")

    # plot color mesh
    if show_colormesh:
        my_cmap = colors.ListedColormap(["b", "r", "g"])
        num_pts = 100
        (wxs, wys) = np.meshgrid(np.linspace(xmin - 1, xmax + 1, num_pts),
                                 np.linspace(ymin - 1, ymax + 1, num_pts))
        wzs = lr.predict(np.concatenate((wxs.reshape(wxs.size, 1),
                                         wys.reshape(wys.size, 1)), axis=1))
        plt.pcolormesh(wxs, wys, wzs.reshape(wxs.shape), cmap=my_cmap, alpha=0.5)

    # plot predicted points
    npred = len(xpred)
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

test5_multiclass()
