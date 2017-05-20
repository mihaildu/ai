"""
Mihail Dunaev
May 2017

Simple test that tries to predict if a tumor is malignant (y = 1)
  or benign (y = 0) based on the tumor size (one feature). It uses
  linear logistic regression to do that. Data is in 'tumors.in' in
  input dir.

For logistic regression I used scikit-learn:
  sklearn.linear_model.LogisticRegression

My matlab result
  x0 = 43.2851
  w = 0.6321, -27.3605
Sklearn result
  C = 1 (lambda = 1 in reg)
    w = 0.0474, -1.42906195
    x0 = w(2)/w(1) ~ 30.1489

  C = 1e5 (lambda = 0.00001)
    w = 0.3360, -14.3312
    x0 = 42.6439

  lr.decision_function is not the sigmoid, it's just a linear function
    it should be 'w(1) * x + w(2) = 0', if so it just compares to 0
    and not 0.5 (instead of computing x0).
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

from common.aux import read_data

def test1_tumors():
    (xs, ys) = read_data("../../../input/tumors.in")

    # train logistic regression model with sklearn
    lr = linear_model.LogisticRegression(C=1e5)
    lr.fit(xs, ys)

    # predict some values
    xpred = np.array([7, 43, 44, 70]).reshape(4, 1);
    ypred = lr.predict(xpred);
    print(ypred.tolist())

    plot_all(xs, ys, lr, xpred, ypred)

def plot_all(xs, ys, lr, xpred, ypred):
    # switch backend - TODO fix this
    plt.switch_backend("TKagg")
    
    N = len(xs)
    xs0 = np.array([xs[i] for i in xrange(N) if ys[i] == 0])
    xs1 = np.array([xs[i] for i in xrange(N) if ys[i] == 1])

    n0 = len(xs0)
    n1 = len(xs1)
    y0 = np.zeros((n0, 1))
    y1 = np.ones((n1, 1))

    # plot initial data
    plt.plot(xs0, y0, "bo")
    plt.plot(xs1, y1, "rx")

    # set range for x/y axis
    xmin = xs.min()
    xmax = xs.max()
    plt.xlim([xmin - 1, xmax + 1])
    plt.ylim([-0.5, 1.5])

    # plot best fit
    num_pts = 100
    wxs = np.linspace(xmin, xmax, num_pts).reshape(num_pts, 1)
    wys = lr.predict_proba(wxs)[:,1]
    plt.plot(wxs, wys, "r-")

    # plot predicted points
    plt.plot(xpred, ypred, color="lime", marker="o", ls="None",
             fillstyle="none")
    plt.show()

test1_tumors()
