"""
Mihail Dunaev
May 2017

Simple test that uses kernel SVM to separate between 2D data
  points (non-linear decision boundary). Data in 'logistic_circle.in'.

For linear SVM I used scikit-learn:
  sklearn.svm.SVC
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib import colors
from matplotlib import cm

from common.aux import read_data

def main():
    """ wrapper for test3_kernel """
    test3_kernel(show_colormesh=True, nice_plotting=True)

def test3_kernel(show_colormesh=True, nice_plotting=True):
    """
    nice_plotting:

      computes the probabilities for SVM as well,
      which makes the training process slower
      (but the plot looks prettier).

      only works with show_colormesh = True.
    """
    (xs, ys) = read_data("../../../input/logistic_circle.in")
    if nice_plotting:
        svc = SVC(C=1e5, kernel="rbf", gamma='auto', probability=True)
    else:
        svc = SVC(C=1e5, kernel="rbf", gamma='auto')
    svc.fit(xs, ys)

    xpred = np.array([3.5, 4, 4, 4.1, 2, 6]).reshape(3, 2)
    ypred = svc.predict(xpred)
    print(ypred.tolist())

    plot_all(xs, ys, svc, xpred, ypred, show_colormesh, nice_plotting)

def plot_all(xs, ys, svc, xpred, ypred,
             show_colormesh=True, nice_plotting=True):

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

    # plot regions
    if show_colormesh:
        num_pts = 100
        my_cmap = colors.ListedColormap(["b", "r"])
        (wxs, wys) = np.meshgrid(np.linspace(xmin - 1, xmax + 1, num_pts),
                                 np.linspace(ymin - 1, ymax + 1, num_pts))
        if nice_plotting:
            wzs = svc.predict_proba(np.concatenate((wxs.reshape(wxs.size, 1),
                                    wys.reshape(wys.size, 1)), axis=1))[:,1]
            plt.imshow(wzs.reshape(wxs.shape), interpolation='bilinear',
                   origin='lower', cmap=my_cmap,
                   extent=(xmin - 1, xmax + 1, ymin - 1, ymax + 1),
                   alpha=0.5)
        else:
            wzs = svc.predict(np.concatenate((wxs.reshape(wxs.size, 1),
                                         wys.reshape(wys.size, 1)), axis=1))
            plt.pcolormesh(wxs, wys, wzs.reshape(wxs.shape), cmap=my_cmap,
                           alpha=0.5)

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
