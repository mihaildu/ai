"""
Mihail Dunaev
May 2017

Simple test that trains a neural net (MLP) to perform logical XNOR.
Arch
  layer 1: x2 x1
  layer 2: n1 n2
  layer 3: n3

For neural network I used scikit-learn:
  sklearn.neural_network.MLPClassifier
"""

import numpy as np
from sklearn.neural_network import MLPClassifier

from common.aux import read_data, plot_all

def test7_xnor2():
    (xs, ys) = read_data("../../../input/xnor_nn.in")

    # train neural net
    nn = MLPClassifier(hidden_layer_sizes=(2), activation="logistic",
                       solver="lbfgs", alpha=1e-5)

    # this performs worse than matlab
    # same trick: train until cost is smaller
    # than threshold or max_iter is reached
    max_iter = 10
    cost_threshold = 0.05
    for i in xrange(max_iter):
        nn.fit(xs, ys)
        if nn.loss_ < cost_threshold:
            break

    # predict some values
    xpred = np.array([0., 0., 0., 1., 1., 0., 1., 1.]).reshape(4, 2);
    ypred = nn.predict(xpred)
    print(ypred)

    plot_all(xs, ys, nn, plot3d=True, colormesh=False)

test7_xnor2()
