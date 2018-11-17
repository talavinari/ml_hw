#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import sgd
from sklearn.preprocessing import normalize

"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""


def perceptron(data, labels):
    """
	returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """
    #TODO check if the zeros is same as nd array of shape 1
    w = np.zeros(shape=(len(data[0])), dtype=float)

    for index, xt in enumerate(data):
        if np.dot(xt, w) >= 0:
            predict = 1
        else:
            predict = -1

        true_label = labels[index]
        if true_label != predict:
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


def exc_a():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = sgd.helper()
    normalized_train_data = normalize_train_date(train_data)

    ns = [5, 10, 50, 100, 500, 1000, 5000]
    results = np.ndarray(shape=(7, 4), dtype=float)
    experiment_times = 100
    for index, n in enumerate(ns):
        accuracies = []
        for i in range(experiment_times):
            array = normalized_train_data[:n]
            sliced_labels = train_labels[:n]
            p = np.random.permutation(len(array))
            array = array[p]
            sliced_labels = sliced_labels[p]
            w = perceptron(array, sliced_labels)
            accuracies.append(calc_accuracy(test_data, test_labels, w))

        results[index] = [n, sum(accuracies) / experiment_times, np.percentile(accuracies, 5),
                          np.percentile(accuracies, 95)]
    print(results)


def normalize_train_date(train_data):
    normalized_train_data = np.ndarray(shape=(len(train_data), len(train_data[0])), dtype=float)
    for index, data in enumerate(train_data):
        normalized_train_data[index] = normalize(data[:, np.newaxis], axis=0).ravel()
    return normalized_train_data


def find_wrong_images(test_data, test_labels, w):
    wrong = []
    for index, test in enumerate(test_data):
        if np.dot(test, w) >= 0:
            predict = 1
        else:
            predict = -1

        if test_labels[index] != predict:
            wrong.append(test)

    return wrong


def exc_b_c_d():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = sgd.helper()
    normalized_train_data = normalize_train_date(train_data)
    w = perceptron(normalized_train_data, train_labels)
    accuracy = calc_accuracy(test_data, test_labels, w)
    print("accuracy of the classifier trained on the full training set applied on the test set " + str(accuracy))

    # plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    # plt.show()
    # wrong = find_wrong_images(test_data, test_labels, w)
    # for wrong_pic in wrong:
    #     plt.imshow(np.reshape(wrong_pic, (28, 28)), interpolation='nearest')
    #     plt.show()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    exc_a()
    exc_b_c_d()

    # print("train data len is " + str(len(train_data)))
    # print("validation data len is " + str(len(validation_data)))
    # print("test data len is " + str(len(test_data)))
    #
    # print("vector size is " + str(len(train_data[0])))
    # print("labels are:" + str(validation_labels))
    # print(max(validation_labels))
    # print(min(validation_labels))
