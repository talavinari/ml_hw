#################################
# Your name:
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.
    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        # TODO check if uniform including/excluding 1
        # TODO check if i should add "check" for case x is exactly 2/4/6/8 and make another binomial draw

        sample_result = np.ndarray(shape=(m, 2), dtype=float)
        x_selection = np.random.uniform(0, 1, m)
        for index, x in enumerate(x_selection):
            if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
                p_to_get_y_1 = 0.8
                y = np.random.binomial(1, p_to_get_y_1)
                sample_result[index] = [x, y]
            else:
                p_to_get_y_1 = 0.1
                y = np.random.binomial(1, p_to_get_y_1)
                sample_result[index] = [x, y]

        return sample_result[sample_result[:, 0].argsort()]

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """

        sample_from_d = self.sample_from_D(m)
        plt.title('Example of ' + str(m) + ' samples , and best intervals with size, k=' + str(k))
        plt.ylabel('value')
        plt.xlabel('x')
        plt.plot(sample_from_d[:, 0], sample_from_d[:, 1], 'ro')
        plt.axis([0, 1, -0.1, 1.1])
        plt.grid(axis='x', linestyle='-')

        erm_intervals, errors = intervals.find_best_interval(sample_from_d[:, 0], sample_from_d[:, 1], k)
        for interval in erm_intervals:
            plt.plot([interval[0], interval[1]], [0, 0], linewidth=2)

        plt.show()

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.
        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        n_steps = (m_last - m_first) / step + 1
        experiment_results = np.ndarray(shape=(int(n_steps), 2), dtype=float)
        result_index = 0
        ms = []

        for m in range(m_first, m_last + 1, step):
            ms.append(m)
            total_empirical_error_rate = 0
            total_true_error_rate = 0
            for t in range(0, T):
                sample = self.sample_from_D(m)
                hypothesis, empirical_errors = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
                true_error = self.calc_true_error(hypothesis)
                total_empirical_error_rate += (empirical_errors / m)
                total_true_error_rate += true_error

            experiment_results[result_index] = [total_empirical_error_rate / T, total_true_error_rate / T]
            result_index += 1

            # print("finished m = " + str(m))

        plt.title('True error (red) & Empirical error (blue) as a function of m')
        plt.ylabel('error')
        plt.xlabel('m - sample size')
        plt.plot(ms, experiment_results[:, 0], 'ro', color='blue')
        plt.plot(ms, experiment_results[:, 1], 'ro', color='red')
        plt.axis([m_first - 1, m_last + 1, 0, 0.6])
        plt.show()
        return experiment_results

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,20.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """

        sample = self.sample_from_D(m)

        steps = (k_last - k_first) / step + 1
        experiment_results = np.ndarray(shape=(int(steps), 3), dtype=float)
        result_index = 0

        for k in range(k_first, k_last + 1, step):
            print("starting k = " + str(k) + "....")
            hypothesis, empirical_errors = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
            true_error = self.calc_true_error(hypothesis)
            print("e(p) = " + str(true_error))
            empirical_error_rate = (empirical_errors / m)
            print("e(s) = " + str(empirical_error_rate))
            experiment_results[result_index] = [k, empirical_error_rate, true_error]
            print(experiment_results[result_index])
            result_index += 1
            print("####################################")

        plt.title('True error (red) & Empirical error (blue) as a function of k')
        plt.ylabel('error')
        plt.xlabel('k - max interval size')
        plt.plot(experiment_results[:, 0], experiment_results[:, 1], 'ro', color='blue')
        plt.plot(experiment_results[:, 0], experiment_results[:, 2], 'ro', color='red')
        plt.axis([k_first - 1, k_last + 1, 0, 0.7])
        plt.show()

        sorted_indices = np.argsort(experiment_results[:, 2])
        lowest_true_error_index = sorted_indices[0]
        return int(experiment_results[lowest_true_error_index][0])

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               k_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """

        sample = self.sample_from_D(m)
        steps = (k_last - k_first) / step + 1
        experiment_results = np.ndarray(shape=(int(steps), 4), dtype=float)
        result_index = 0

        for k in range(k_first, k_last + 1, step):
            print("starting k = " + str(k) + "....")
            hypothesis, empirical_errors = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
            empirical_error_rate = (empirical_errors / m)
            print("e(s) = " + str(empirical_error_rate))
            penalty = self.calc_penalty(2 * k, m, 0.1)
            print("penalty is = " + str(penalty))
            print("penalty + e(s) = " + str(penalty + empirical_error_rate))
            experiment_results[result_index] = [k, empirical_error_rate, penalty, penalty + empirical_error_rate]
            print(experiment_results[result_index])
            result_index += 1

        plt.title('Empirical error (blue), penalty (red), e(s) + penalty (green) as a function of k')
        plt.ylabel('error')
        plt.xlabel('k - max interval size')
        plt.plot(experiment_results[:, 0], experiment_results[:, 1], color='blue')
        plt.plot(experiment_results[:, 0], experiment_results[:, 2], color='red')
        plt.plot(experiment_results[:, 0], experiment_results[:, 3], color='green')
        plt.axis([k_first - 1, k_last + 1, 0, 3])
        plt.show()

        sorted_indices = np.argsort(experiment_results[:, 3])
        lowest_penalty_with_empirical_error = sorted_indices[0]
        return int(experiment_results[lowest_penalty_with_empirical_error][0])

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        sample = self.sample_from_D(m)
        experiment_results = []
        ks = []
        for k in range(1, 11):
            ks.append(k)
            total_errors = 0
            print("running with k=" + str(k))
            holdout_size = 0
            for t in range(0, T):
                train_sample, holdout_samples = self.split_sample_for_cross(sample, t)
                holdout_size = len(holdout_samples)
                hypothesis, empirical_errors = intervals.find_best_interval(train_sample[:, 0], train_sample[:, 1], k)
                for test in holdout_samples:
                    if self.predict(hypothesis, test[0]) != test[1]:
                        total_errors += 1

            holdout_error = (total_errors / T) / holdout_size
            print("AVG errors after 3 runs is : " + str(holdout_error))
            experiment_results.append(holdout_error)

        plt.title('Avg e(hold_out) as a function of best h in k size interval')
        plt.ylabel('avg e(hold_out)')
        plt.xlabel('ERM result of max k intervals hypothesis class')
        plt.plot(ks, experiment_results, 'ro', color='blue')
        plt.axis([0, 11, 0, 1])
        plt.show()

        return np.argmin(experiment_results) + 1

    def get_prob_to_get_value_1(self, interval):
        if not self.is_in_one_section(interval):
            raise ValueError('cant get prob for given interval')
        if interval[1] <= 0.2:
            return 0.8
        elif interval[0] >= 0.2 and interval[1] <= 0.4:
            return 0.1
        elif interval[0] >= 0.4 and interval[1] <= 0.6:
            return 0.8
        elif interval[0] >= 0.6 and interval[1] <= 0.8:
            return 0.1
        elif interval[0] >= 0.8 and interval[1] <= 1:
            return 0.8
        else:
            print("error, invalid interval")
            return None

    def is_in_one_section(self, interval):
        if interval[1] <= 0.2:
            return True
        elif interval[0] >= 0.2 and interval[1] <= 0.4:
            return True
        elif interval[0] >= 0.4 and interval[1] <= 0.6:
            return True
        elif interval[0] >= 0.6 and interval[1] <= 0.8:
            return True
        elif interval[0] >= 0.8 and interval[1] <= 1:
            return True
        else:
            return False

    def split_interval(self, interval):
        new_intervals = []
        is_one_section = self.is_in_one_section(interval)
        if is_one_section:
            new_intervals.append(interval)
        else:
            cut_place = 0
            if interval[0] < 0.2 and interval[1] > 0.2:
                new_intervals.append((interval[0], 0.2))
                cut_place = 0.2
            elif interval[0] < 0.4 and interval[1] > 0.4:
                new_intervals.append((interval[0], 0.4))
                cut_place = 0.4
            elif interval[0] < 0.6 and interval[1] > 0.6:
                new_intervals.append((interval[0], 0.6))
                cut_place = 0.6
            elif interval[0] < 0.8 and interval[1] > 0.8:
                new_intervals.append((interval[0], 0.8))
                cut_place = 0.8
            new_intervals.extend(self.split_interval((cut_place, interval[1])))

        return new_intervals

    def split_intervals(self, intervals):
        new_intervals = []
        interval_prediction = []
        prev = 0
        for interval in intervals:
            # handle 0 prediction between the hypothesis intervals
            splitted_intervals = self.split_interval((prev, interval[0]))
            new_intervals.extend(splitted_intervals)
            for x in range(len(splitted_intervals)):
                interval_prediction.append(0)

            # handle ERM intervals
            splitted_intervals = self.split_interval(interval)
            new_intervals.extend(splitted_intervals)
            for x in range(len(splitted_intervals)):
                interval_prediction.append(1)

            prev = interval[1]

        # handle final hypothesis interval to number - 1
        splitted_intervals = self.split_interval((intervals[len(intervals) - 1][1], 1))
        new_intervals.extend(splitted_intervals)
        for x in range(len(splitted_intervals)):
            interval_prediction.append(0)

        return new_intervals, interval_prediction

    def calc_true_error(self, h):
        new_intervals, interval_prediction = self.split_intervals(h)
        true_error = 0
        for i in range(0, len(new_intervals)):
            interval = new_intervals[i]
            prob_to_interval = interval[1] - interval[0]
            prob_to_1 = self.get_prob_to_get_value_1(interval)

            # the interval stands for 0 prediction
            if interval_prediction[i] == 0:
                true_error += (prob_to_interval * prob_to_1)
            # the interval stands for 1 prediction
            else:
                true_error += (prob_to_interval * (1 - prob_to_1))
        return true_error

    def split_sample_for_cross(self, original_sample, seed):
        size_of_hold_out = int(0.2 * len(original_sample))
        max_number = len(original_sample) - 1
        holdout_random_indices = np.random.RandomState(seed).choice(max_number, size_of_hold_out, replace=False)

        holdout_samples = np.ndarray(shape=(size_of_hold_out, 2), dtype=float)
        train_samples = np.ndarray(shape=(len(original_sample) - size_of_hold_out, 2), dtype=float)

        holdout_index = 0
        train_index = 0
        for index, sample in enumerate(original_sample):
            if index in holdout_random_indices:
                holdout_samples[holdout_index] = original_sample[index]
                holdout_index += 1
            else:
                train_samples[train_index] = original_sample[index]
                train_index += 1

        return train_samples, holdout_samples

    def predict(self, hypothesis, x):
        for interval in hypothesis:
            if interval[0] <= x <= interval[1]:
                return 1

        return 0

    def calc_penalty(self, vcdim, n, delta):
        return np.sqrt((32 / n) * ((vcdim * (np.log((2 * np.e * n) / vcdim))) + np.log((4 / delta))))


if __name__ == '__main__':
    ass = Assignment2()
    # plt.plot([1, 2, 3, 4, 5,6,7,8], [9, 7.5, 5, 3,1.4,0.8,0.4,0.3], color='blue')
    # plt.axis([1, 9, 1, 10])
    # plt.show()
    # ass.draw_sample_intervals(100, 3)
    # ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    # ass.experiment_k_range_erm(1500, 1, 10, 1)
    print("best k is : " + str(ass.experiment_k_range_srm(1500, 1, 10, 1)))
    # ass.cross_validation(1500, 3)
