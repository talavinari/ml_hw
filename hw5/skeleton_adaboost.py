#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
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
        if x[self.index] > self.threshold:
            return self.predict_value
        return self.predict_value * -1

    def calc_error(self, X, Y, D):
        error = 0
        for x, y, d in zip(X, Y, D):
            if self.predict(x) != y:
                error += d

        return error


def get_WL(current_distribution, X_train, y_train):
    max_occur_per_word = X_train.max(axis=0)
    predict_options = [1, -1]
    best_h = None
    best_error = 1
    for predict in predict_options:
        for index in range(len(X_train[0])):
            for theta in range(int(max_occur_per_word[index])):
                h = Hypothesis(theta, predict, index)
                error = h.calc_error(X_train, y_train, current_distribution)
                # print("h with index: " + str(index) + ", theta:" + str(theta) + ", prediction:" + str(predict))
                # print("error is " + str(error))
                if error < best_error:
                    best_error = error
                    best_h = h

    return best_h, best_error


def predict(h, x):
    if x[h[1]] > h[2]:
        return h[0]
    return h[0] * -1
    pass


def calc_error(X_train, y_train, hypotheses, alpha_vals):
    errors = 0
    for x, y in zip(X_train, y_train):
        sign = 0
        for alpha, h in zip(alpha_vals, hypotheses):
            sign += alpha * predict(h, x)

        if (sign >= 0 and y == -1) or (sign < 0 and y == 1):
            errors += 1

    return errors / len(X_train)

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
    # TODO: add your code here

    hypotheses = []
    alpha_vals = []
    n = len(X_train)
    current_distribution = [1 / n] * n

    for t in range(T):
        h_t, e_t = get_WL(current_distribution, X_train, y_train)
        alpha_t = 0.5 * np.log((1 - e_t) / e_t)
        z_t = 2 * np.sqrt(e_t * (1 - e_t))
        for i in range(len(current_distribution)):
            value_to_exponent = alpha_t
            if h_t.predict(X_train[i]) != y_train[i]:
                value_to_exponent *= -1
            current_distribution[i] = (current_distribution[i] * np.exp(value_to_exponent)) / z_t

        hypotheses.append((h_t.predict_value, h_t.index, h_t.threshold))
        alpha_vals.append(alpha_t)
        print("iteration " + str(t))

        print("train error : " + str(calc_error(X_train, y_train, hypotheses, alpha_vals)))

    return hypotheses, alpha_vals


##############################################
# You can add more methods here, if needed.


##############################################


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    print("vocabulary length")
    print(len(vocab))
    print("train length")
    print(len(X_train))

    print("test length")
    print(len(X_test))

    # print(my_tuple)
    # print((X_train, y_train, X_test, y_test, vocab))

    hypotheses, alpha_vals = run_adaboost(X_train, y_train, 80)

    ##############################################
    # You can add more methods here, if needed.

    ##############################################


if __name__ == '__main__':
    main()
