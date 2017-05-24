"""
Mihail Dunaev
May 2017

Simple test that performs Linear Discriminant Analysis (LDA)
  on 2D data (2D -> 1D). Input in 'lda.in'.

Results
  Matlab
    LDA mat = [-0.6159, 0.7878]
    slope = -1.2791
  Sklearn
    LDA mat = [2.5618, -2.3366]
    slope = -0.9121

For LDA I used scikit-learn:
  sklearn.discriminant_analysis.LinearDiscriminantAnalysis
  this also has a 'predict()' function implemented
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from common.aux import read_data

def lda_test():
    (xs, ys) = read_data("../../../input/lda.in")

    # feature scaling and mean norm
    nxs = scale(xs)

    # compute proj matrix
    # this time it's called lda.scalings_
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(nxs, ys)

    # project input on new dimension
    zs = lda.transform(nxs)

    # plot everything
    plot_all(nxs, ys, zs, lda)

def plot_all(xs, ys, zs, lda):
    # switch backend - TODO fix this
    plt.switch_backend("TKagg")

    # plot initial data
    N = len(xs)
    xs0 = np.array([xs[i] for i in xrange(N) if ys[i] == 0])
    xs1 = np.array([xs[i] for i in xrange(N) if ys[i] == 1])
    plt.plot(xs0[:,0], xs0[:,1], "bo")
    plt.plot(xs1[:,0], xs1[:,1], "rx")

    # plot projected points
    # TODO there is something wrong with lda.scalings_
    # it does the job but it doesn't project like
    # I would expect; investigate
    pxs = np.dot(zs, lda.scalings_.T)
    pxs0 = np.array([pxs[i] for i in xrange(N) if ys[i] == 0])
    pxs1 = np.array([pxs[i] for i in xrange(N) if ys[i] == 1])
    plt.plot(pxs0[:,0], pxs0[:,1], "bo")
    plt.plot(pxs1[:,0], pxs1[:,1], "ro")

    # set range for x/y axis
    xmin = pxs[:,0].min()
    xmax = pxs[:,0].max()
    ymin = pxs[:,1].min()
    ymax = pxs[:,1].max()
    plt.xlim([xmin - 1, xmax + 1])
    plt.ylim([ymin - 1, ymax + 1])

    # plot lda line
    wxs = np.array([[xmin - 1], [xmax + 1]])
    slope = lda.scalings_[1][0] / lda.scalings_[0][0]
    wys = wxs * slope
    plt.plot(wxs, wys, color="lime")
    plt.show()

lda_test()
