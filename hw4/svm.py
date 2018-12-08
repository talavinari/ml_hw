#################################
# Your name: Tal Avinari
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""


# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    results = np.ndarray(shape=(3, 2), dtype=float)

    clf = svm.SVC(C=1000.0, decision_function_shape='ovr', kernel='linear')
    clf.fit(X_train, y_train)
    results[0] = clf.n_support_

    # create_plot(X_train, y_train, clf)
    # plt.show()

    clf = svm.SVC(C=1000.0, decision_function_shape='ovr', kernel='poly', degree=2)
    clf.fit(X_train, y_train)
    results[1] = clf.n_support_

    # create_plot(X_train, y_train, clf)
    # plt.show()

    clf = svm.SVC(C=1000.0, decision_function_shape='ovr', kernel='rbf')
    clf.fit(X_train, y_train)
    results[2] = clf.n_support_

    # create_plot(X_train, y_train, clf)
    # plt.show()

    return results


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    validation_size = len(X_val)
    results = np.ndarray(shape=(11,), dtype=float)

    log_search = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    for index, log in enumerate(log_search):
        current_c = pow(10, log)
        clf = svm.SVC(C=current_c, decision_function_shape='ovr', kernel='linear')
        clf.fit(X_train, y_train)
        predict = clf.predict(X_val)
        correct_classifications = np.count_nonzero(predict == y_val)
        accuracy = correct_classifications / validation_size
        results[index] = accuracy

        # print("accuracy for c=" + str(current_c) + " is :" + str(accuracy))
        # if current_c == 100:
        #     create_plot(X_train, y_train, clf)
        #     plt.show()

    # plt.title('Accuracy on the validation set, as a function of C')
    # plt.ylabel('Accuracy')
    # plt.xlabel('log10(C)')
    # plt.plot(log_search, results, color='blue')
    # plt.axis([log_search[0], log_search[len(log_search) - 1], 0.3, 1.1])
    # plt.grid(axis='x', linestyle='-')
    # plt.grid(axis='y', linestyle='-')
    # plt.show()

    return results


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    validation_size = len(X_val)
    results = np.ndarray(shape=(11,), dtype=float)

    log_search = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    for index, log in enumerate(log_search):
        gamma = pow(10, log)
        clf = svm.SVC(C=10, decision_function_shape='ovr', kernel='rbf', gamma=gamma)
        clf.fit(X_train, y_train)
        predict = clf.predict(X_val)
        correct_classifications = np.count_nonzero(predict == y_val)
        accuracy = correct_classifications / validation_size
        results[index] = accuracy
        # print("accuracy for gamma=" + str(gamma) + " is :" + str(accuracy))
        # if gamma == 10:
        #     create_plot(X_train, y_train, clf)
        #     plt.show()

    # plt.title('RBF accuracy on the validation set, as a function of gamma')
    # plt.ylabel('RBF accuracy')
    # plt.xlabel('log10(gamma)')
    # plt.plot(log_search, results, color='blue')
    # plt.axis([log_search[0], log_search[len(log_search) - 1], 0.3, 1.1])
    # plt.grid(axis='x', linestyle='-')
    # plt.grid(axis='y', linestyle='-')
    # plt.show()

    return results


if __name__ == '__main__':
    training_data, training_labels, validation_data, validation_labels = get_points()
    train_three_kernels(training_data, training_labels, validation_data, validation_labels)
    linear_accuracy_per_C(training_data, training_labels, validation_data, validation_labels)
    rbf_accuracy_per_gamma(training_data, training_labels, validation_data, validation_labels)
