"""
Mihail Dunaev
May 2017

Simple test that uses linear SVM to separate between 2D data
  points. Data in 'logistic.in'.

For linear SVM I used scikit-learn:
  sklearn.svm.LinearSVC (liblinear in the back)
  sklearn.svm.SVC (libsvm in the back)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC

from common.aux import read_data

def main():
    """ wrapper for test2_linear """
    test2_linear(use_liblinear=True)

def test2_linear(use_liblinear=True):
    (xs, ys) = read_data("../../../input/logistic.in")

    # train support vector machine classifier
    if use_liblinear:
        svc = LinearSVC(C=1e10)
    else:
        # TODO C doesn't seem to work, investigate
        svc = SVC(C=1e5, kernel="linear")

    svc.fit(xs, ys)

    # predict some values
    xpred = np.array([3.5, 4, 4, 4.1, 2, 6]).reshape(3, 2)
    ypred = svc.predict(xpred)
    print(ypred.tolist())

    plot_all(xs, ys, svc, xpred, ypred, show_lines=True)

def plot_all(xs, ys, svc, xpred, ypred, show_lines=False):
    # switch backend - TODO fix this
    plt.switch_backend("TKagg")

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

    # plot separating lines
    if show_lines:
        # w(1) * x1 + w(2) * x2 + w(3)
        w1 = svc.coef_[0][0]
        w2 = svc.coef_[0][1]
        w3 = svc.intercept_[0]
        x0 = np.array([[xmin], [xmax]])
        y0normal = np.array([[-(w3 + w1 * xmin) / w2],
                             [-(w3 + w1 * xmax) / w2]])
        y0plus = np.array([[1 - (w3 + w1 * xmin) / w2],
                        [1 - (w3 + w1 * xmax) / w2]])
        y0minus = np.array([[-1 - (w3 + w1 * xmin) / w2],
                        [-1 - (w3 + w1 * xmax) / w2]])
        plt.plot(x0, y0normal, "g-")
        plt.plot(x0, y0plus, "r-")
        plt.plot(x0, y0minus, "b-")

    # plot predicted points
    npred = xpred.shape[0]
    for i in xrange(npred):
        if ypred[i] == 0:
            plt.plot(xpred[i][0], xpred[i][1], "bs")
        elif ypred[i] == 1:
            plt.plot(xpred[i][0], xpred[i][1], "rs")
        else:
            plt.plot(xpred[i][0], xpred[i][1], "ks")
    plt.show()

main()
