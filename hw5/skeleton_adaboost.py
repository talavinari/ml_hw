#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


class Hypothesis:
    threshold = 5

    def predict(self, x):
        if x > self.threshold:
            return 1
        return -1


def calc_error(hypothesis, X, Y, D):
    error = 0
    for x, y, d in zip(X, Y, D):
        if hypothesis.predict(x) != y:
            error += d

    return error


def get_WL(current_distribution, X_train, y_train):
    return Hypothesis()


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

    for t in T:
        h_t = get_WL(current_distribution, X_train, y_train)
        e_t = calc_error(h_t, X_train, y_train, current_distribution)
        alpha_t = 0.5 * np.log((1 - e_t) / e_t)
        z_t = 2 * np.sqrt(e_t * (1 - e_t))
        for i in range(len(current_distribution)):
            value_to_exponent = alpha_t
            if h_t.predict(X_train[i]) != y_train:
                value_to_exponent *= -1
            current_distribution[i] = (current_distribution[i] * np.exp(value_to_exponent)) / z_t

        hypotheses.append((1, 2, 3))
        alpha_vals.append(alpha_t)

    return hypothesis


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

    # hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    ##############################################
    # You can add more methods here, if needed.

    ##############################################


if __name__ == '__main__':
    main()
