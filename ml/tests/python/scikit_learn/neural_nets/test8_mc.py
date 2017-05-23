"""
Mihail Dunaev
May 2017

Simple test that trains a neural net to separate between
  4 classes of points in 2D. Data is in 'xnor_mc_nn.in'.
Arch
  2, 2, 4

Results
  Max number of iterations: 3219 (c = 0.1195)

For neural network I used scikit-learn:
  sklearn.neural_network.MLPClassifier
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.neural_network import MLPClassifier

from common.aux import read_data

def test8_mc():
    (xs, ys) = read_data("../../../input/xnor_mc_nn.in", output_layers=4)
    #fit_and_plot_cost(xs,ys)

    # train neural net
    nn = MLPClassifier(hidden_layer_sizes=(2), activation="logistic",
                       solver="lbfgs", alpha=1e-5)

    # train several times (seems to work best)
    # other solution: use max_iter=5000 (doesn't work all the time)
    max_iter = 10
    cost_threshold = 0.05
    for i in xrange(max_iter):
        nn.fit(xs, ys)
        if nn.loss_ < cost_threshold:
            break

    # predict some values
    xpred = np.array([0.1, 0.1, -0.1, 0.9, 0.9, 0.2, 0.8, 1.1, 0.4, 0.5,
                      0.7, 0.25, 0.25, 0.55, 0.5, 0.5]).reshape(8, 2);
    ypred = nn.predict(xpred)
    print(ypred)

    my_plot(xs, ys, nn, xpred, ypred, colormesh=True, show_unknown=True)

def fit_and_plot_cost(xs, ys):
    """
    Fits a neural network to (xs, ys)
      and plots evolution of cost.

    For some reason it always reaches the threshold.
    """
    nn = MLPClassifier(hidden_layer_sizes=(2), activation="logistic",
                       solver="lbfgs", alpha=1e-5, max_iter=1,
                       warm_start=True)
    # save cost vals
    max_iter = 200
    cost_vals = []
    cost_threshold = 0.05
    for i in xrange(max_iter):
        nn.fit(xs, ys)
        cost_vals.append(nn.loss_)
        if nn.loss_ < cost_threshold:
            break

    # plot cost
    plt.switch_backend("TKagg")
    plt.grid(True)
    plt.xlim([0, len(cost_vals) + 1])
    plt.ylim([0, max(cost_vals)])
    plt.plot(range(1, len(cost_vals) + 1), cost_vals, "b-")
    plt.show()

def my_plot(xs, ys, nn, xpred, ypred, colormesh=False, show_unknown=True):
    """
    show_unknown = shows gray area
      where all probs are small (< 0.5)
      only works with 'colormesh'
    """
    # switch backend - TODO fix this
    plt.switch_backend("TKagg")

    # plot initial data
    N = len(xs)
    for i in xrange(N):
        if ys[i][0] == 1:
            plt.plot(xs[i][0], xs[i][1], "bo")
        elif ys[i][1] == 1:
            plt.plot(xs[i][0], xs[i][1], "rs")
        elif ys[i][2] == 1:
            plt.plot(xs[i][0], xs[i][1], "gd")
        elif ys[i][3] == 1:
            plt.plot(xs[i][0], xs[i][1], "y^")
        else:
            plt.plot(xs[i][0], xs[i][1], "kx")

    # set range for x/y axis
    xmin = xs[:,0].min()
    xmax = xs[:,0].max()
    ymin = xs[:,1].min()
    ymax = xs[:,1].max()
    plt.xlim([xmin - 1, xmax + 1])
    plt.ylim([ymin - 1, ymax + 1])

    # plot predicted values
    npred = len(xpred)
    for i in xrange(npred):
        if ypred[i][0] == 1:
            plt.plot(xpred[i][0], xpred[i][1], "bx")
        elif ypred[i][1] == 1:
            plt.plot(xpred[i][0], xpred[i][1], "rx")
        elif ypred[i][2] == 1:
            plt.plot(xpred[i][0], xpred[i][1], "gx")
        elif ypred[i][3] == 1:
            plt.plot(xpred[i][0], xpred[i][1], "yx")
        else:
            plt.plot(xpred[i][0], xpred[i][1], "kx")

    # plot colormesh
    if colormesh:
        num_pts = 100
        (wxs, wys) = np.meshgrid(np.linspace(xmin - 1, xmax + 1, num_pts),
                                 np.linspace(ymin - 1, ymax + 1, num_pts))
        if show_unknown:
            my_cmap = colors.ListedColormap(["b", "r", "g", "y", "k"])
            wzs_pred = nn.predict(np.concatenate((wxs.reshape(wxs.size, 1),
                     wys.reshape(wys.size, 1)), axis=1))
            wzs = np.zeros((wxs.size, 1))
            for i in xrange(wxs.size):
                if wzs_pred[i].max() == 0:
                    wzs[i] = 4
                else:
                    wzs[i] = wzs_pred[i].argmax()
        else:
            my_cmap = colors.ListedColormap(["b", "r", "g", "y"])
            wzs = nn.predict_proba(np.concatenate((wxs.reshape(wxs.size, 1),
                     wys.reshape(wys.size, 1)), axis=1)).argmax(axis=1)
        plt.pcolormesh(wxs, wys, wzs.reshape(wxs.shape),
                           cmap=my_cmap, alpha=0.5)
    plt.show()

test8_mc()
