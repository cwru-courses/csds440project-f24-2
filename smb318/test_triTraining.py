from unittest import TestCase
import numpy as np
from numpy.ma.testutils import assert_array_equal
from sklearn.tree import DecisionTreeClassifier

from triTraining import triTraining


class Test(TestCase):
    def test_bootstrap_sample(self):
        data = np.array([1, 2, 3, 4, 5])
        labels = np.array([10, 20, 30, 40, 50])
        sampled_data, sampled_labels = triTraining.bootstrap_sample(data, labels)

        self.assertEqual(len(sampled_data), len(data))
        self.assertEqual(len(sampled_labels), len(labels))

        # checks that sampled indices exist in the original data
        for val in sampled_data:
            self.assertIn(val, data)

    # dummy classifier for testing, always predicts 1
    class dummy1:
        @staticmethod
        def predict(data):
            return np.array([1] * len(data))

    # another dummy classifier, alternates returning 0 and 1
    class dummy2:
        @staticmethod
        def predict(data):
            return np.array([i % 2 for i in range(len(data))])

    # tests agreement_indices and measure_error
    def test_agreement_indices_measure_error(self):
        # case where classifiers always agree
        hj = Test.dummy1()
        hk = Test.dummy1()

        data = np.array([0, 1, 2, 3, 4])
        labels = np.array([1, 1, 1, 1, 1])  # labels match what classifier predicts

        assert_array_equal(triTraining.agreement_indices(hj, hk, data), [0, 1, 2, 3, 4])
        self.assertEqual(triTraining.measure_error(hj, hk, data, labels), 0.0)

        # case where classifiers never agree. should return 0 by convention
        hj = Test.dummy1()
        hk = Test.dummy2()

        data = np.array([0])
        labels = np.array([1])

        assert_array_equal(triTraining.agreement_indices(hj, hk, data), [])
        self.assertEqual(triTraining.measure_error(hj, hk, data, labels), 0.0)

        # case where classifiers sometimes agree, and there are some joint misclassifications
        hj = Test.dummy1()
        hk = Test.dummy2()

        data = np.array([0, 1, 2, 3, 4])
        labels = np.array([1, 1, 1, 0, 0])

        assert_array_equal(triTraining.agreement_indices(hj, hk, data), [1, 3])
        self.assertEqual(triTraining.measure_error(hj, hk, data, labels), 0.5)

    def test_subsample(self):
        data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        labels = np.array([0, 1, 0, 1])

        with self.assertRaises(AssertionError) as context:
            triTraining.subsample(data, labels, len(labels) + 1)
        self.assertEqual(str(context.exception), "n cannot be greater than the number of examples")

        sub_data, sub_labels = triTraining.subsample(data, labels, len(labels))

        for d, l in zip(sub_data, sub_labels):
            self.assertIn(d, data)
            self.assertIn(l, labels)

    def test_measure_errors(self):
        base_learner = DecisionTreeClassifier()
        model = triTraining(base_learner)

        # uses different max_depths so that the joint misclassifications are different for each pair of classifiers
        model.learners = [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=1),
                          DecisionTreeClassifier()]

        data = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        labels = np.array([0, 1, 1, 0])

        for learner in model.learners:
            learner.fit(data, labels)

        errors = model.measure_errors(data, labels)

        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            self.assertEqual(errors[i], triTraining.measure_error(model.learners[j], model.learners[k], data, labels))
