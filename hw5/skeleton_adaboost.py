#################################
# Your name:
#################################

import math

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np

from process_data import parse_data

np.random.seed(7)


class Hypothesis:

    def __init__(self, threshold, predict_value, index):
        self.threshold = threshold
        self.predict_value = predict_value
        self.index = index

    def predict(self, x):
        if x[self.index] <= self.threshold:
            return self.predict_value
        return self.predict_value * -1


def WL(current_distribution, X_train, y_train):
    index_positive, tetha_positive, e_t_positive = ERMforDecisionStumps(current_distribution, X_train, y_train)
    index_negative, tetha_negative, e_t_negative = ERMforDecisionStumps(current_distribution, X_train, y_train * -1)

    if e_t_positive < e_t_negative:
        return index_positive, tetha_positive, e_t_positive, 1
    else:
        return index_negative, tetha_negative, e_t_negative, -1


def ERMforDecisionStumps(current_distribution, X_train, y_train):
    F_star = math.inf

    for j in range(len(X_train[0])):
        column_values = X_train[:, j]
        # argsort_by_x = np.argsort(column_values)
        # sorted_y = y_train[argsort_by_x]
        # distribution_by_sort = current_distribution[argsort_by_x]
        # sorted_x = sorted(column_values)
        # sorted_x.append(sorted_x[len(sorted_x) - 1] + 1)
        #
        all_data = list(zip(column_values, y_train, current_distribution))
        sorted_data = sorted(all_data, key=lambda item: item[0])

        # F = sum(distribution_by_sort[i] for i in range(len(X_train)) if sorted_y[i] == 1)
        F = sum(w for x, y, w in sorted_data if y == 1)

        if F < F_star:
            F_star = F
            # tetha_star = sorted_x[0] - 1
            tetha_star = sorted_data[0][0] - 1
            j_star = j
        for i in range(len(X_train)):
            # F = F - sorted_y[i] * distribution_by_sort[i]
            F = F - sorted_data[i][1] * sorted_data[i][2]
            # if F < F_star and sorted_x[i] != sorted_x[i + 1]:
            if i + 1 == len(X_train):
                next_x = sorted_data[i][0] + 1
            else:
                next_x = sorted_data[i + 1][0]
            if F < F_star and sorted_data[i][0] != next_x:
                F_star = F
                # tetha_star = 0.5 * (sorted_x[i] + sorted_x[i + 1])
                tetha_star = 0.5 * (sorted_data[i][0] + next_x)
                j_star = j

    return j_star, tetha_star, F_star


def predict(h, x):
    if x[h[1]] <= h[2]:
        return h[0]
    return h[0] * -1


def calc_error(Xs, labels, hypotheses, alpha_vals, t):
    errors = 0
    for x, y in zip(Xs, labels):
        sign = 0
        for i in range(t + 1):
            sign += alpha_vals[i] * predict(hypotheses[i], x)
        # for alpha, h in zip(alpha_vals, hypotheses):
        #     sign += alpha * predict(h, x)

        if sign * y < 0:
            errors += 1

    return errors / len(Xs)


def run_adaboost(X_train, y_train, T):
    """
    Returns:

        hypotheses :
            A list of T tuples describing the hypotheses chosen by the algorithm.
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals :
            A list of T float values, which are the alpha values obtained in every
            iteration of the algorithm.
    """
    hypotheses = []
    alpha_vals = []
    n = len(X_train)
    current_distribution = [1 / n] * n
    current_distribution = np.array(current_distribution)

    for t in range(T):
        index, tetha_star, e_t, predict_value = WL(current_distribution, X_train, y_train)
        print("iteration " + str(t))
        h_t = Hypothesis(tetha_star, predict_value, index)
        alpha_t = 0.5 * np.log((1 - e_t) / e_t)
        print("hypothesis for T=" +
              str(t) + " is"
              + " index=" + str(index)
              + ", threshold=" + str(tetha_star)
              + ", predict=" + str(predict_value)
              + ", alpha=" + str(alpha_t))

        z_t = 2 * np.sqrt(e_t * (1 - e_t))
        for i in range(len(current_distribution)):
            value_to_exponent = -1 * alpha_t * y_train[i] * h_t.predict(X_train[i])
            current_distribution[i] = (current_distribution[i] * np.exp(value_to_exponent)) / z_t

        hypotheses.append((h_t.predict_value, h_t.index, h_t.threshold))
        alpha_vals.append(alpha_t)

        print("train error : " + str(calc_error(X_train, y_train, hypotheses, alpha_vals, t)))

    return hypotheses, alpha_vals


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    T = 5
    experiment_results = np.ndarray(shape=(int(T), 3), dtype=float)
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    for i in range(T):
        train_error = calc_error(X_train, y_train, hypotheses, alpha_vals, i)
        test_error = calc_error(X_test, y_test, hypotheses, alpha_vals, i)
        experiment_results[i] = [i, train_error, test_error]

    plt.title('Test error (red) & Train error (blue) as a function of T')
    plt.ylabel('error')
    plt.xlabel('T')
    plt.plot(experiment_results[:, 0], experiment_results[:, 1], linestyle='dashdot', color='blue')
    plt.plot(experiment_results[:, 0], experiment_results[:, 2], linestyle='dashdot', color='red')
    plt.axis([0, T, 0, 0.4])
    plt.show()


if __name__ == '__main__':
    main()
