#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import matplotlib.pyplot as plt

C_TYPE = 'C'
N_TYPE = 'N'

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

    for t in range(1, T + 1):
        nt = eta_0 / t
        i = np.random.randint(len(data))
        yi = labels[i]
        xi = data[i]
        if yi * np.dot(w, xi) < 1:
            w = np.add(np.dot(1 - nt, w), np.dot(nt * C * yi, xi))
        else:
            w = np.dot(1 - nt, w)

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
    avg_accuracies, log_search = calc_parameter(N_TYPE, 1)

    plt.title('Average accuracy on the validation set, as a function of n0')
    plt.ylabel('average accuracy')
    plt.xlabel('log10(n0)')
    plt.plot(log_search, avg_accuracies, color='blue')
    plt.axis([log_search[0], log_search[len(log_search) - 1], 0.45, 1])
    plt.grid(axis='x', linestyle='-')
    plt.grid(axis='y', linestyle='-')
    plt.show()


def exc_b():
    avg_accuracies, log_search = calc_parameter(C_TYPE, 1)

    plt.title('Average accuracy on the validation set, as a function of C')
    plt.ylabel('average accuracy')
    plt.xlabel('log10(C)')
    plt.plot(log_search, avg_accuracies, color='blue')
    plt.axis([log_search[0], log_search[len(log_search) - 1], 0.45, 1])
    plt.grid(axis='x', linestyle='-')
    plt.grid(axis='y', linestyle='-')
    plt.show()

    print(max(avg_accuracies))


def calc_parameter(param_type, second_as_const):
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    log_search = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    experiment_times = 10
    avg_accuracies = []
    for log in log_search:
        total_accuracy = 0
        for t in range(experiment_times):
            if param_type == N_TYPE:
                classifier = SGD(train_data, train_labels, second_as_const, pow(10, log), 1000)
            else:
                classifier = SGD(train_data, train_labels, pow(10, log), second_as_const, 1000)
            accuracy = calc_accuracy(validation_data, validation_labels, classifier)
            total_accuracy += accuracy

        avg_accuracy = total_accuracy / experiment_times
        avg_accuracies.append(avg_accuracy)
        print("accuracy for 10^" + str(log) + " is :" + str(avg_accuracy))
    return avg_accuracies, log_search


def exc_c_d():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    classifier = SGD(train_data, train_labels, pow(10, -4), 1, 20000)
    accuracy = calc_accuracy(test_data, test_labels, classifier)
    print("accuracy for c=10^-4 and n0=1 is : " + str(accuracy))
    plt.imshow(np.reshape(classifier, (28, 28)), interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    exc_a()
    exc_b()
    exc_c_d()
