#################################
# Your name:
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):

    def is_in_one_section(self, interval):
        if interval[1] <= 0.2:
            return True, 1
        elif interval[0] >= 0.2 and interval[1] <= 0.4:
            return True, 0
        elif interval[0] >= 0.4 and interval[1] <= 0.6:
            return True, 1
        elif interval[0] >= 0.6 and interval[1] <= 0.8:
            return True, 0
        elif interval[0] >= 0.8 and interval[1] <= 1:
            return True, 1
        else:
            return False, -1

    def split_interval(self, interval):
        new_intervals = []
        is_one_section, section = self.is_in_one_section(interval)
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
        interval_value = []
        prev = 0
        for interval in intervals:
            # handle 0 place between the hypothesis intervals
            splitted_intervals = self.split_interval((prev, interval[0]))
            new_intervals.extend(splitted_intervals)
            for x in range(len(splitted_intervals)):
                interval_value.append(0)

            # handle ERM intervals
            splitted_intervals = self.split_interval(interval)
            new_intervals.extend(splitted_intervals)
            for x in range(len(splitted_intervals)):
                interval_value.append(1)

            prev = interval[1]

        # handle final hypothesis interval to 1
        splitted_intervals = self.split_interval((intervals[len(intervals) - 1][1], 1))
        new_intervals.extend(splitted_intervals)
        for x in range(len(splitted_intervals)):
            interval_value.append(0)

        return new_intervals, interval_value

    """Assignment 2 skeleton.



    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """


    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        # TODO: Implement me

        # TODO check if uniform including/excluding 1
        # todo check if i should add "check" for case x is exactly 2/4/6/8 and make another binomial draw
        y_values = []
        x_selection = np.random.uniform(0, 1, m)
        for x in x_selection:
            if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
                p_to_get_y_1 = 0.8
                y = np.random.binomial(1, p_to_get_y_1)
                y_values.append(y)
            else:
                p_to_get_y_1 = 0.1
                y = np.random.binomial(1, p_to_get_y_1)
                y_values.append(y)

        x_arg_sort = np.argsort(x_selection)
        y_sorted = []
        for x in x_arg_sort:
            y_sorted.append(y_values[x])

        arr = np.ndarray(shape=(2, m), dtype=float)
        arr[0] = np.sort(x_selection)
        arr[1] = y_sorted

        return arr


    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        arr = self.sample_from_D(m)
        # plt.plot(arr[0], arr[1], 'ro')
        # plt.axis([0, 1, -0.1, 1.1])
        # plt.grid(axis='x', linestyle='-')
        # plt.show()

        best = intervals.find_best_interval(arr[0], arr[1], k)

        # print(best)
        print("best k is " + str(len(best[0])))

        fig, ax = plt.subplots()
        for interval in best[0]:
            ax.plot([interval[0], interval[1]], [0, 0], linewidth=2)
            print(interval)

        plt.axis([0, 1, -0.1, 1.1])
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
        # TODO: Implement the loop

        for m in range(m_first, m_last + 1, step):
            arr = self.sample_from_D(m)
            h, empirical_errors = intervals.find_best_interval(arr[0], arr[1], k)
            print("error count is " + str(empirical_errors))
            print("e(s) = " + str(empirical_errors / m))
            print(h)
            # print(ass.split_interval(interval))
            new_intervals, interval_value = self.split_intervals(h)
            print(new_intervals)
            print(interval_value)

            # prob_to_error_in_section_1 = 0.2
            # prob_to_error_in_section_0 = 0.1
            # prev_interval_end = 0
            # real_error = 0
            # # split_intervals(h)
            # for interval in h:
            #     is_inside, section_type = self.is_in_one_section(interval)
            #     if is_inside:
            #         prob = interval[1] - interval[0]
            #         if section_type == 1:
            #             real_error += prob * prob_to_error_in_section_1
            #         else:
            #             real_error += prob * prob_to_error_in_section_0
            #     else:
            #         print("error, found bigger then one section interval, interval:")
            #         print(interval)
            #     prev_interval_end = interval[1]


    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,20.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        # TODO: Implement the loop
        pass


    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        # TODO: Implement the loop
        pass


    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # TODO: Implement me
        pass


#################################
# Place for additional methods

#################################


if __name__ == '__main__':
    ass = Assignment2()
    # ass.draw_sample_intervals(100, 3)
    # ass.experiment_m_range_erm(10, 100, 5, 3, 100)

    ass.experiment_m_range_erm(10, 20, 5, 3, 100)
    print(ass.split_interval((0.399, 0.99999)))

    # ass.experiment_k_range_erm(1500, 1, 20, 1)
    # ass.experiment_k_range_srm(1500, 1, 20, 1)
    # ass.cross_validation(1500, 3)
