#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import sgd
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle

"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""


def perceptron(data, labels):
    """
	returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """

    w = np.zeros(shape=(len(data[0])), dtype=float)

    for index, xt in enumerate(data):
        if np.dot(xt, w) >= 0:
            predict = 1
        else:
            predict = -1

        true_label = labels[index]
        if true_label != predict:
            # print("bad prediction - prediction is " + str(predict) + " real label is" + str(true_label))
            x = np.dot(xt, true_label)
            w = np.add(w, x)

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
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = sgd.helper()
    print("train data len is " + str(len(train_data)))
    print("validation data len is " + str(len(validation_data)))
    print("test data len is " + str(len(test_data)))

    print("vector size is " + str(len(train_data[0])))
    print("labels are:" + str(validation_labels))
    print(max(validation_labels))
    print(min(validation_labels))

    normalized_train_data = np.ndarray(shape=(len(train_data), len(train_data[0])), dtype=float)
    for index, data in enumerate(train_data):
        normalized_train_data[index] = normalize(data[:, np.newaxis], axis=0).ravel()

    ns = [5, 10, 50, 100, 500, 1000, 5000]
    results = np.ndarray(shape=(7, 2), dtype=float)

    experiment_times = 100

    for index, n in enumerate(ns):
        total_accuracy = 0
        for i in range(experiment_times):
            array = normalized_train_data[:n]
            sliced_labels = train_labels[:n]
            p = np.random.permutation(len(array))
            array = array[p]
            sliced_labels = sliced_labels[p]
            w = perceptron(array, sliced_labels)
            accuracy = calc_accuracy(test_data, test_labels, w)
            total_accuracy += accuracy

        results[index] = [n, total_accuracy / experiment_times]

    print(results)
