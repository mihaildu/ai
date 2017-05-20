"""
Mihail Dunaev
May 2017

Simple test that tries to separate 2 sets of 2D data on a plane
  using polynomial (2nd degree) logistic regression. Data is in 
  'logistic_circle.in'.

For logistic regression I used scikit-learn:
  sklearn.linear_model.LogisticRegression
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from scipy.special import expit
from mpl_toolkits.mplot3d import Axes3D

from common.aux import read_data

def test4_circle():
    (xs, ys) = read_data("../../../input/logistic_circle.in")

    # train logistic regression model with sklearn
    lr = linear_model.LogisticRegression(C=1e5)
    lr.fit(np.concatenate((xs ** 2, xs), axis=1), ys)

    # predict some values
    xpred = np.array([3.5, 4, 4, 4.1, 2, 6]).reshape(3, 2);
    ypred = lr.predict(np.concatenate((xpred ** 2, xpred), axis=1))
    print(ypred.tolist())

    plot_all(xs, ys, lr, xpred, ypred, plot3d=True)

def plot_all(xs, ys, lr, xpred, ypred, plot3d=False):
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
    if plot3d:
        num_pts = 100
        xvals = np.linspace(xmin, xmax, num_pts)
        yvals = np.linspace(ymin, ymax, num_pts)
        (wxs, wys) = np.meshgrid(xvals, yvals)
        # w(1) * x1 ^ 2 + w(2) * x2 ^ 2 + w(3) * x1 + w(4) * x2 + w(5)
        w1 = lr.coef_[0][0]
        w2 = lr.coef_[0][1]
        w3 = lr.coef_[0][2]
        w4 = lr.coef_[0][3]
        w5 = lr.intercept_[0]
        wzs = np.array(map(expit, (wxs ** 2) * w1 + (wys ** 2) * w2 +
                           wxs * w3 + wys * w4 + w5))
        ax.plot_surface(wxs, wys, wzs)

    # plot predicted points
    if plot3d:
        ax.plot(xpred[:,0], xpred[:,1], ypred, color="lime", marker="o",
                ls="None")
    else:
        npred = len(xpred)
        for i in xrange(npred):
            if ypred[i] == 0:
                plt.plot(xpred[i][0], xpred[i][1], "bs")
            elif ypred[i] == 1:
                plt.plot(xpred[i][0], xpred[i][1], "rs")
            else:
                plt.plot(xpred[i][0], xpred[i][1], "ks")
    plt.show()

test4_circle()
