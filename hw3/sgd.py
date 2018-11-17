#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = 0, 8
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the final classifier
    """

    w = np.zeros(shape=(len(data[0])), dtype=float)

    for t in range(T):
        i = np.random.randint(len(data))
        yi = labels[i]
        xi = data[i]
        if yi * np.dot(w, xi) < 1:
            w = np.add(np.dot(w, 1 - eta_0), np.dot(xi, eta_0 * yi * C))
        else:
            w = np.dot(w, 1 - eta_0)

    return w


#################################

# Place for additional code

#################################

def calc_accuracy(test_data, test_labels, w):
    mistake = 0

    for index, xt in enumerate(test_data):
        if np.dot(xt, w) >= 0:
            predict = 1
        else:
            predict = -1

        true_label = test_labels[index]
        if true_label != predict:
            mistake += 1

    test_size = len(test_data)
    return (test_size - mistake) / test_size


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()

    # log_search = [-5]
    log_search = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    experiment_times = 10

    for log in log_search:
        total_accuracy = 0
        for t in range(experiment_times):
            classifier = SGD(train_data, train_labels, 1, pow(10, log), 1000)
            accuracy = calc_accuracy(validation_data, validation_labels, classifier)
            total_accuracy += accuracy

        avg_accuracy = total_accuracy / experiment_times
        print("accuracy for  10^" + str(log) + " is: " + str(avg_accuracy))
