"""
Mihail Dunaev
May 2017

Simple test that tries to predict house prices using linear
  regression. There is only one feature (house size) and data
  is stored in 'houses.in' in input dir.

For linear regression I used scikit-learn:
  sklearn.linear_model.LinearRegression

My matlab result
  w = 10.8619, -22.2128
Sklearn result
  w(1) = lr.coef_ = 10.86193234
  w(2) = lr.intercept_ = -22.21280098
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict

def test1():
    (xs, ys) = read_data("../../../input/houses.in")

    # train linear regression model with sklearn
    lr = linear_model.LinearRegression()
    lr.fit(xs, ys)

    # predict some values
    xpred = np.array([12, 45]).reshape(2, 1)
    ypred = lr.predict(xpred)
    print(ypred.tolist())

    # predicting with cross_val_predict()
    # this will try to split xs into xtrain/xcv (xtest too?)
    # and then predict for each xs
    # e.g. xs[0] will be in cv (with other pts)
    # then xs[1], xs[2] ... and so on
    # TODO check if this tries to fit again
    #ypred = cross_val_predict(lr, xs, ys, cv=10)
    #print(ypred.tolist())

    #w = np.array([[10.8619], [-22.2128]])
    #old_plot(xs, ys, w)
    plot_all(xs, ys, lr)

def plot_all(xs, ys, lr):
    # switch backend - TODO fix this
    plt.switch_backend("TKagg")
    plt.figure(1)

    # plot initial data
    plt.plot(xs, ys, "bo")

    # plot best fit
    xmin = xs.min()
    xmax = xs.max()
    wxs = np.array([[xmin], [xmax]])
    wys = lr.predict(wxs)
    plt.plot(wxs, wys, "r-")
    plt.show()

def read_data(fname):
    with open(fname, "r") as f:
        lines = f.readlines()
        N = len(lines)
        xs = np.zeros((N, 1))
        ys = np.zeros((N, 1))
        for line, i in zip(lines, xrange(N)):
            vals = line.split()
            xs[i] = int(vals[0])
            ys[i] = int(vals[1])
    return (xs, ys)

def old_plot(xs, ys, w):
    # switch backend - TODO fix this
    plt.switch_backend("TKagg")
    plt.figure(2)

    # plot initial data
    plt.plot(xs, ys, "bo")

    # plot best fit
    xmin = xs.min()
    xmax = xs.max()
    wxs = np.array([[xmin], [xmax]])
    wys = np.dot(np.concatenate((wxs, np.ones((2, 1))), axis=1), w)
    plt.plot(wxs, wys, "r-")
    plt.show(block=False)

test1()
