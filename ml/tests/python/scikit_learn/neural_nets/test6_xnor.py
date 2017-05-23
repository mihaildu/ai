"""
Mihail Dunaev
May 2017

Simple test that builds an XNOR neural net
  and predicts some values.
  layer 1: x2 x1
  layer 2: n1 n2
  layer 3: n3

XNOR values
  0 xnor 0 = 1
  0 xnor 1 = 0
  1 xnor 0 = 0
  1 xnor 1 = 1

For neural network I used scikit-learn:
  sklearn.neural_network.MLPClassifier
"""

import numpy as np
from sklearn.neural_network import MLPClassifier

from common.aux import read_data, plot_all

def test6_xnor():
    nn = MLPClassifier(hidden_layer_sizes=(2), activation="logistic",
                       solver="lbfgs", warm_start=True)

    # set weights
    nn.coefs_ = [np.array([[20, -20],[20, -20]]), np.array([[20], [20]])]
    nn.intercepts_ = [np.array([-30, 10]), np.array([-10])]

    # fill in some stuff that are normally added at nn.fit()
    nn.classes_ = np.array([0, 1])
    nn.n_layers_ = 3
    nn.n_outputs_ = 1
    nn.out_activation_ = "logistic"
    nn.n_iter_ = 1

    # there are more stuff to add, like _label_binarizer
    # since I don't want to add these myself, I'll start
    # a new fit with the current weights (warm_start=True)
    # this should add the missing stuff and keep the arch
    # since we are already at a minimum, fit() should stop
    # right away
    (xs, ys) = read_data("../../../input/xnor_nn.in")
    nn.fit(xs, ys)

    xpred = np.array([0., 0., 0., 1., 1., 0., 1., 1.]).reshape(4, 2);
    ypred = nn.predict(xpred)
    print(ypred)

    # extra plot of the network
    plot_all(xs, ys, nn, plot3d=True, colormesh=False)

test6_xnor()
