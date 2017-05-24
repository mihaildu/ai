"""
Mihail Dunaev
May 2017

Simple test that uses linear SVM to separate between 1D data
  points. Data in 'tumors.in'.

For linear SVM I used scikit-learn:
  sklearn.svm.LinearSVC (liblinear in the back)
  sklearn.svm.SVC (libsvm in the back)
  SVC seems to perform better than LinearSVC
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC

from common.aux import read_data

def main():
    """ wrapper for test1_linear """
    test1_linear(use_liblinear=False)

def test1_linear(use_liblinear=True):
    (xs, ys) = read_data("../../../input/tumors.in")

    # train support vector machine classifier
    if use_liblinear:
        svc = LinearSVC(C=1e10, tol=1e-20)
    else:
        svc = SVC(C=1e5, kernel="linear")

    svc.fit(xs, ys)

    # predict some values
    xpred = np.array([7, 43, 44, 70]).reshape(4, 1);
    ypred = svc.predict(xpred);
    print(ypred.tolist())

    plot_all(xs, ys, svc, xpred, ypred, show_line=False)

def plot_all(xs, ys, svc, xpred, ypred, show_line=False):
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
    plt.ylim([-1.0, 1.5])

    # plot decision line (wt * x = 0)
    if show_line:
        wxs = np.array([[xmin], [xmax]])
        wys = np.dot(np.array([[xmin, 1], [xmax, 1]]),
                     np.array([[svc.coef_[0][0]], [svc.intercept_[0]]]))
        plt.plot(wxs, wys, "r-")

    # plot predicted points
    plt.plot(xpred, ypred, color="lime", marker="o", ls="None",
             fillstyle="none")
    plt.show()

main()
