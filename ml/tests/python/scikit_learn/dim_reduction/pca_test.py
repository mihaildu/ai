"""
Mihail Dunaev
May 2017

Simple test that performs Principal Component Analysis (PCA)
  on 2D data (2D -> 1D). Input in 'logistic.in'.

Results
  Matlab
    PCA mat = [-0.8454, -0.5342]
    slope = 0.6319
    variance retained = 90.04%
  Sklearn
    PCA mat = [0.7071, 0.7071]
    slope = 0.99 (not identical vals)
    variance retained = 88.51%

For PCA I used scikit-learn:
  sklearn.decomposition.PCA
  this uses LAPACK for svd
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

from common.aux import read_data

def pca_test():
    (xs, _) = read_data("../../../input/logistic.in")

    # feature scaling and mean norm
    nxs = scale(xs)

    # compute proj matrix
    # this is actually pca.components_
    # or pca.components_.T for transpose
    pca = PCA(n_components=1)
    pca.fit(nxs)
    print("Variance retained: %.2f%%" % (pca.explained_variance_ratio_ * 100))

    # project input on new dimension
    zs = pca.transform(nxs)

    # plot everything
    plot_all(nxs, zs, pca)

def plot_all(xs, zs, pca):
    # switch backend - TODO fix this
    plt.switch_backend("TKagg")

    # plot initial data
    plt.plot(xs[:,0], xs[:,1], "ko")

    # set range for x/y axis
    xmin = xs[:,0].min()
    xmax = xs[:,0].max()
    ymin = xs[:,1].min()
    ymax = xs[:,1].max()
    plt.xlim([xmin - 1, xmax + 1])
    plt.ylim([ymin - 1, ymax + 1])

    # plot pca line
    wxs = np.array([[xmin - 1], [xmax + 1]])
    slope = pca.components_[0][1] / pca.components_[0][0]
    wys = wxs * slope
    plt.plot(wxs, wys, color="lime")

    # plot projected points
    # this is the same as pca.inverse_transform(zs)
    pxs = np.dot(zs, pca.components_)
    plt.plot(pxs[:,0], pxs[:,1], "ro")
    plt.show()

pca_test()
