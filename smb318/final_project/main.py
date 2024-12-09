import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

from modifiedTriTraining import modifiedTriTraining
from triTraining import triTraining


# fetches the datasets and returns their features and labels
def preprocessed_datasets():
    # fetch datasets
    ionosphere = fetch_ucirepo(id=52)
    australia = fetch_ucirepo(id=143)

    ionosphere_features = np.array(ionosphere.data.features)
    australia_features = np.array(australia.data.features)

    ionosphere_labels = np.array(ionosphere.data.targets).flatten()  # flatten() makes 1d
    ionosphere_labels = np.where(ionosphere_labels == 'g', 1, 0)  # encodes 'g' as 1, 'b' as 0

    australia_labels = np.array(australia.data.targets).flatten()

    return ionosphere_features, ionosphere_labels, australia_features, australia_labels


# returns the error of a tri training classifier, either triTraining or modifiedTriTraining
def classifier_error(classifier, X: np.ndarray, y: np.ndarray, unlabel_rate: float) -> float:
    # keeps 25% of data as test examples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled = train_test_split(
        X_train, y_train, test_size=unlabel_rate, random_state=42)

    classifier.fit(X_train_labeled, y_train_labeled, X_train_unlabeled)
    predictions = classifier.predict(X_test)

    return 1 - accuracy_score(predictions, y_test)


ionosphere_features, ionosphere_labels, australia_features, australia_labels = preprocessed_datasets()

# combinations of classifiers used for modified tri training
combo1 = [DecisionTreeClassifier(), LogisticRegression(), KNeighborsClassifier()]
combo2 = [SVC(), GaussianNB(), DecisionTreeClassifier()]
combo3 = [SVC(), DecisionTreeClassifier(), LogisticRegression()]
combo4 = [DecisionTreeClassifier(), KNeighborsClassifier(), GaussianNB()]
combos = [combo1, combo2, combo3, combo4]


# returns the mean error of a tri training classifier (using 10 tries) on a given dataset, given an unlabel rate
def mean_triTraining_error(X: np.ndarray, y: np.ndarray, classifier, unlabel_rate):
    errors = np.array([classifier_error(classifier, X, y, unlabel_rate) for _ in range(10)])
    return np.mean(errors)


# compares the performance of regular and modified training on some dataset, using the above combinations of classifiers
# base learner is the base learner we are using for regular tri training
def compare_performance(dataset_name: str, X: np.ndarray, y: np.ndarray, base_learner):
    tri_errors = []
    combo_errors = {f"combo{i}_error": [] for i in range(1, 5)}

    u_rates = [0.8, 0.6, 0.4, 0.2]

    # stores errors
    for u_rate in u_rates:
        tri_error = mean_triTraining_error(X, y, triTraining(base_learner), u_rate)
        tri_errors.append(tri_error)

        combo_errors_list = np.array([
            classifier_error(modifiedTriTraining(combo), X, y, u_rate)
            for combo in combos
        ])
        for i, error in enumerate(combo_errors_list):
            combo_errors[f"combo{i + 1}_error"].append(error)

    plt.figure(figsize=(10, 6))
    plt.plot(u_rates, tri_errors, marker='o', label='TriTraining Error')

    for combo_name, errors in combo_errors.items():
        plt.plot(u_rates, errors, marker='o', label=combo_name)

    plt.title(f"Error Rates vs. Unlabel Rate, {dataset_name}")
    plt.xlabel('Unlabel Rate')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.show()


compare_performance("Australia dataset", australia_features, australia_labels, DecisionTreeClassifier())
compare_performance("ionosphere dataset", ionosphere_features, ionosphere_labels, DecisionTreeClassifier())
