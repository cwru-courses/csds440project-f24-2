import numpy as np
import random
import math
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from typing import Tuple, List


class triTraining:

    # for a dataset of size n, samples n examples uniformly w/ replacement
    @staticmethod
    def bootstrap_sample(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(data)
        indices = np.random.randint(0, n, size=n)
        return data[indices], labels[indices]

    # returns the indices in a dataset where classifiers c1 and c2 agree
    @staticmethod
    def agreement_indices(c1, c2, data: np.ndarray) -> np.ndarray:
        pred_c1 = c1.predict(data)
        pred_c2 = c2.predict(data)
        return np.where(pred_c1 == pred_c2)[0]

    # approximates classification error of hypothesis by dividing the number of labeled examples on which both c1 and c2
    # make incorrect classification by the number of labeled examples on which the classifiers agree
    @staticmethod
    def measure_error(c1, c2, data: np.ndarray, labels: np.ndarray) -> float:
        agreements = triTraining.agreement_indices(c1, c2, data)

        # if there are no agreements, return 0 by convention
        if len(agreements) == 0:
            return 0.0

        predictions = c1.predict(data)[agreements]

        # error rate
        return 1 - accuracy_score(labels[agreements], predictions)

    # randomly chooses n rows from data and labels. returns the examples and corresponding labels
    @staticmethod
    def subsample(data: np.ndarray, labels: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        num_examples = len(labels)

        assert n <= num_examples, "n cannot be greater than the number of examples"
        indices = random.sample(range(num_examples), n)
        return data[indices], labels[indices]

    # base learner is the base classifier we want to use, such as DecisionTreeClassifier
    def __init__(self, base_learner):
        self.base_learner = base_learner
        self.learners = [clone(base_learner) for _ in range(3)]

    def measure_errors(self, data: np.ndarray, labels: np.ndarray) -> List[float]:
        e = [None, None, None]
        for i in range(3):
            # j and k are the other 2 classifiers
            j = (i + 1) % 3
            k = (i + 2) % 3
            e[i] = triTraining.measure_error(self.learners[j], self.learners[k], data, labels)

        return e

    # L_data is the set of labeled examples (features), L_labels is the set of corresponding labels. U_data is a set
    # of unlabeled examples
    def fit(self, L_data: np.ndarray, L_labels: np.ndarray, U_data: np.ndarray):
        e_prime = [0.5, 0.5, 0.5]  # classifier error rates
        l_prime = [0.0, 0.0, 0.0]

        # initializes learners
        for i in range(3):
            sample_data, sample_labels = triTraining.bootstrap_sample(L_data, L_labels)
            self.learners[i].fit(sample_data, sample_labels)

        # don't have to recalculate each time since we only use the original labeled examples
        e = self.measure_errors(L_data, L_labels)

        L_new_data = [np.empty((0, 0)) for _ in range(3)]  # new examples that we will add to each classifier
        L_new_labels = [np.empty(0) for _ in range(3)]  # new labels that we will add to each classifier
        update = [True, True, True]  # whether we need to update each classifier

        while any(update):
            for i in range(3):
                L_new_data[i] = np.empty((0, 0))
                L_new_labels[i] = np.empty(0)
                update[i] = False

                if e[i] < e_prime[i]:
                    # self.learners[j] and self.learners[k] are the other 2 classifiers
                    j = (i + 1) % 3
                    k = (i + 2) % 3

                    # adds unlabeled examples for which the other 2 classifiers agree on the label
                    agreements = triTraining.agreement_indices(self.learners[j], self.learners[k], U_data)
                    new_examples = U_data[agreements]
                    L_new_data[i] = new_examples
                    L_new_labels[i] = self.learners[j].predict(new_examples)

                    if l_prime[i] == 0:  # the current classifier hasn't been updated in this case
                        l_prime[i] = e[i] / (e_prime[i] - e[i]) + 1

                    # this is before subsample, so we cannot reuse num_new_examples after the following loop
                    num_new_examples = len(L_new_labels)
                    if l_prime[i] < num_new_examples:
                        if e[i] * num_new_examples < e_prime[i] * l_prime[i]:
                            update[i] = True
                        elif l_prime[i] > (e[i] / (e_prime[i] - e[i])):
                            L_new_data[i], L_new_labels[i] = triTraining.subsample(L_new_data[i], L_new_labels[i],
                                                                                   math.ceil(e_prime[i] * l_prime[i] /
                                                                                             e[i] - 1))
                            update[i] = True

            for i in range(3):
                if update[i]:
                    data = np.concatenate((L_data, L_new_data[i]), axis=0)
                    labels = np.concatenate((L_labels, L_new_labels[i]), axis=0)
                    self.learners[i].fit(data, labels)
                    e_prime[i] = e[i]
                    l_prime[i] = len(L_new_labels)

    def predict(self, data: np.ndarray):
        predictions = np.array([learner.predict(data) for learner in self.learners])

        # gets majority vote for each data point
        majority_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

        return majority_predictions
