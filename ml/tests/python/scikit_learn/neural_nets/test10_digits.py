"""
Mihail Dunaev
May 2017

Use neural net to recognize digit in image.

Info
  Input/xtrain
    5000 images 20x20 pixels, grayscale
    each image has a digit drawn
    500 images for each digit

  Arch
    400 features, 25 hidden units, 10 output units

  Output/ytrain
    class for the digit in the image (10 classes)
    class 10 = digit 0

  Stats
    Matlab
      training takes 28.29min
    Sklearn
      training takes 8-9s

  Results (accuracy)
    Matlab
      nn - 99.48% (weights_test6_nn.mat)
      logreg - 94.42% (weights_test6_log.mat)
      nn with 70%/30% - 91.33% (weights_test6_nn_70.mat)
    Sklearn
      nn - 99.92% (weights_test6_nn_py)
      nn with 70%/30% - 86.40% (weights_test6_nn_70_py)

For neural network I used scikit-learn:
  sklearn.neural_network.MLPClassifier
"""

import cv2
import time
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from scipy.io import loadmat

def main():
    """ wrapper for test10 """
    test10_digits(display_input=False, train=False, show_accuracy=True,
                  use_test=False, display_predicted=True)

def test10_digits(display_input=False, train=False,
                  store_weights_to_file=False, show_accuracy=True,
                  use_test=False, display_predicted=True,
                  store_fname="../../../input/weights_test6_nn_py",
                  load_fname="../../../input/weights_test6_nn_py",
                  input_fname="../../../input/digits_nn.mat"):
    """
    Input
      display_input: if True, some input images will be displayed
        to the user; default is 5 for each digits.
      train: if True the program will try to train the neural net;
        if False the weights are loaded from a file.
      store_weights_to_file: if True weights will be stored to file;
        only works when train is set to True.
      show_accuracy: computes accuracy for test data; if only train
        data is used, it will be computed for all train data.
      use_test: if True, input data will be split into train and test;
        right now the split is hardcoded at 70% train, 30% test.
      display_predicted: if True, a number of randomly selected images
        will be shown along with their prediction.
      store_fname = file name where to store weights.
      load_fname = file name from where to load the weights.
      input_fname = file name from where to read the input (train & test)
        data.
    """
    if use_test:
        (xtrain, ytrain, xtest, ytest) = my_read_data_split(input_fname)
    else:
        (xtrain, ytrain) = my_read_data(input_fname)

    # show how input data looks like
    if display_input:
        print("Showing train images")
        display_data(xtrain, num_images=5, delta_images=500)

    if train:
        # train neural net
        nn = MLPClassifier(hidden_layer_sizes=(25), activation="logistic",
                       solver="lbfgs", alpha=1e-5)
        start_time = time.time()
        nn.fit(xtrain, ytrain)
        print("Total training time: %.2fs" % (time.time() - start_time))
        if store_weights_to_file:
            store_weights(store_fname, nn)
    else:
        # load weights from file
        nn = load_weights(load_fname)

    # test prediction
    if show_accuracy:
        if use_test:
            acc = compute_accuracy(xtest, ytest, nn)
        else:
            acc = compute_accuracy(xtrain, ytrain, nn)
        print("Accuracy: %.2f%%" % acc)

    # display some images with prediction
    if display_predicted:
        if use_test:
            display_with_predict(xtest, nn, num_images=10)
        else:
            display_with_predict(xtrain, nn, num_images=10)

def display_with_predict(xs, nn, num_images=10):
    """
    Shows 'num_images' images and prints their prediction.
    """
    N = xs.shape[0]
    perm = np.random.permutation(N)
    cv2.namedWindow("Digit", cv2.WINDOW_NORMAL)
    for k in xrange(num_images):
        index = perm[k]
        img = xs[index].reshape((20, 20), order="F")
        ypred = nn.predict(xs[index].reshape(1, -1))
        if ypred.max() == 0:
            print("Unable to predict digit")
        else:
            digit = ypred.argmax() + 1
            if digit == 10:
                digit = 0
            print("Predicted digit is %d" % digit)

        cv2.imshow("Digit", np.interp(img, [-1,1], [0,1]))
        cv2.waitKey(0)

def store_weights(fname, nn):
    """
    Not enough to store nn.coefs_ &
    intercepts_. Store entire nn since
    it's easier.
    """
    with open(fname, "w") as f:
        pickle.dump(nn, f)

def load_weights(fname):
    """
    Returns neural net
    (with weights) from fname.
    """
    with open(fname, "r") as f:
        return pickle.load(f)

def compute_accuracy(xs, ys, nn):
    """
    Computes accuracy for xs with the neural
    network nn. Actual values are in ys.

    This might be improved to count as correct
    even if the prediction is unknown, but argmax()
    shows the correct digit.
    """
    num_total = xs.shape[0]
    num_correct = 0.0
    for k in xrange(num_total):
        ypred = nn.predict(xs[k].reshape(1, -1))
        if ((ypred.max() == 0 and ys[k].max() == 0) or
            (ypred.argmax() == ys[k].argmax())):
            num_correct = num_correct + 1.0
    return num_correct * 100 / num_total

def display_data(xs, num_images, delta_images):
    """
    Shows 'num_images' images every 'delta_images'.
    """
    N = xs.shape[0]
    cv2.namedWindow("Digit", cv2.WINDOW_NORMAL)
    for i in xrange(0, N, delta_images):
        for j in xrange(num_images):
            img = xs[i+j].reshape((20, 20), order="F")
            cv2.imshow("Digit", np.interp(img, [-1,1], [0,1]))
            cv2.waitKey(0)

def my_read_data_split(fname):
    """ Reads train & test data sets """
    # 70% = 3500, 30% = 1500
    Ntrain = 3500
    Ntest = 1500
    num_classes = 10
    num_features = 400
    limit_train = Ntrain / num_classes
    limit_test = Ntest / num_classes

    # pre-allocate data
    xtrain = np.zeros((Ntrain, num_features))
    ytrain = np.zeros((Ntrain, num_classes))
    xtest = np.zeros((Ntest, num_features))
    ytest = np.zeros((Ntest, num_classes))
    index_train = 0
    index_test = 0
    cnt = 0

    turn = "train"
    data = loadmat(fname)
    num_inputs = data["X"].shape[0]
    for i in xrange(num_inputs):
        if turn == "train":
            xtrain[index_train] = data["X"][i]
            j = data["y"][i]
            ytrain[index_train][j-1] = 1
            index_train = index_train + 1
            cnt = cnt + 1
            if cnt == limit_train:
                turn = "test"
                cnt = 0
        else:
            xtest[index_test] = data["X"][i]
            j = data["y"][i]
            ytest[index_test][j-1] = 1
            index_test = index_test + 1
            cnt = cnt + 1
            if cnt == limit_test:
                turn = "train"
                cnt = 0

    return (xtrain, ytrain, xtest, ytest)

def my_read_data(fname):
    """ Reads train data set """
    data = loadmat(fname)
    xs = data["X"]
    num_classes = 10
    num_inputs = len(xs)
    ys = np.zeros((num_inputs, num_classes))
    for i in xrange(num_inputs):
        j = data["y"][i]
        ys[i][j-1] = 1
    return (xs, ys)

main()
