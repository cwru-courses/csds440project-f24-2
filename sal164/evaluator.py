import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from linear_subspace import LinearSubspaceModel


class Evaluator:
    """
    Evaluates the performance of the Linear Subspace Model on arbitrary datasets.
    """

    def __init__(self, model=None, pca_components=10, logistic_max_iter=1000):
        """
        Initializes the evaluator with a model, PCA components, and logistic regression settings.
        
        Args:
            model: An instance of LinearSubspaceModel.
            pca_components: Number of components for PCA.
            logistic_max_iter: Maximum iterations for logistic regression.
        """
        self.model = model or LinearSubspaceModel(m=10, p=5, ridge_alpha=1.0)
        self.pca_components = pca_components
        self.logistic_max_iter = logistic_max_iter

    def create_views(self, X_train, X_test, method="pca"):
        """
        Creates two views from the data using the specified method.
        
        Args:
            X_train: Training data.
            X_test: Test data.
            method: The method to create views ('pca', 'random_projection', 'kernel_pca', 
                    'autoencoder', 'clustering', 'statistical_transform').
        
        Returns:
            X1_train, X2_train, X1_test, X2_test: The two views for the training and test datasets.
        """
        if method == "pca":
            pca = PCA(n_components=self.pca_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            X1_train, X2_train = X_train, X_train_pca
            X1_test, X2_test = X_test, X_test_pca
        
        elif method == "random_projection":
            rp = GaussianRandomProjection(n_components=self.pca_components)
            X_train_rp = rp.fit_transform(X_train)
            X_test_rp = rp.transform(X_test)
            X1_train, X2_train = X_train, X_train_rp
            X1_test, X2_test = X_test, X_test_rp
        
        elif method == "kernel_pca":
            kpca = KernelPCA(n_components=self.pca_components, kernel='rbf')
            X_train_kpca = kpca.fit_transform(X_train)
            X_test_kpca = kpca.transform(X_test)
            X1_train, X2_train = X_train, X_train_kpca
            X1_test, X2_test = X_test, X_test_kpca
        
        elif method == "autoencoder":
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            autoencoder = MLPRegressor(hidden_layer_sizes=(self.pca_components * 2, self.pca_components), 
                                       max_iter=500)
            autoencoder.fit(X_train_scaled, X_train_scaled)
            
            X_train_ae = autoencoder.predict(X_train_scaled)
            X_test_ae = autoencoder.predict(X_test_scaled)
            X1_train, X2_train = X_train, X_train_ae
            X1_test, X2_test = X_test, X_test_ae
        
        elif method == "clustering":
            kmeans = KMeans(n_clusters=self.pca_components, random_state=42)
            X_train_cluster_labels = kmeans.fit_predict(X_train)
            X_test_cluster_labels = kmeans.predict(X_test)
            X_train_cluster_view = np.eye(self.pca_components)[X_train_cluster_labels]
            X_test_cluster_view = np.eye(self.pca_components)[X_test_cluster_labels]
            X1_train, X2_train = X_train, X_train_cluster_view
            X1_test, X2_test = X_test, X_test_cluster_view

        elif method == "statistical_transform":
            # Example of a simple statistical transformation: square root and log transformations
            X_train_sqrt = np.sqrt(np.abs(X_train) + 1e-8)
            X_train_log = np.log(np.abs(X_train) + 1e-8 + 1)
            X_test_sqrt = np.sqrt(np.abs(X_test) + 1e-8)
            X_test_log = np.log(np.abs(X_test) + 1e-8 + 1)
            X1_train, X2_train = X_train_sqrt, X_train_log
            X1_test, X2_test = X_test_sqrt, X_test_log

        elif method == 'feature_split':
            # Split feature space in half
            mid_index = X_train.shape[1] // 2
            X1_train, X2_train = X_train[:, :mid_index], X_train[:, mid_index:]
            X1_test, X2_test = X_test[:, :mid_index], X_test[:, mid_index:]
        
        else:
            raise ValueError(f"Unknown method '{method}' for creating views.")
        
        return X1_train, X2_train, X1_test, X2_test

    def train_and_evaluate(self, X1_train, X2_train, y_train, X1_test, X2_test, y_test, label_fraction=0.1):
        """
        Trains the Linear Subspace Model and evaluates it using logistic regression.
        """
        # Split a small portion of labeled data for supervised training
        n_labeled = int(len(X1_train) * label_fraction)
        X1_train_small, y_train_small = X1_train[:n_labeled], y_train[:n_labeled]
        X2_train_small = X2_train[:n_labeled]

        self.model.fit(X1_train, X2_train)
        X_train_transformed = self.model.transform(X1_train, X2_train)
        X_train_small_transformed = self.model.transform(X1_train_small, X2_train_small)
        X_test_transformed = self.model.transform(X1_test, X2_test)
        
        clf = LogisticRegression(max_iter=self.logistic_max_iter)
        clf.fit(X_train_small_transformed, y_train_small)
        y_pred_test = clf.predict(X_test_transformed)
        accuracy_transformed = accuracy_score(y_test, y_pred_test)
        
        log_reg = LogisticRegression(max_iter=self.logistic_max_iter)
        log_reg.fit(X1_train_small, y_train_small)
        y_pred_log_reg = log_reg.predict(X1_test)
        accuracy_original = accuracy_score(y_test, y_pred_log_reg)
        
        return {
            'accuracy_transformed': accuracy_transformed,
            'accuracy_original': accuracy_original
        }

    def evaluate(self, X, y, test_size=0.3, random_state=42, label_fraction=0.05, method='pca'):
        """
        Full evaluation pipeline, including data splitting, view creation, model training, and evaluation.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X1_train, X2_train, X1_test, X2_test = self.create_views(X_train, X_test, method=method)
        results = self.train_and_evaluate(X1_train, X2_train, y_train, X1_test, X2_test, y_test, label_fraction)
        return results


# Example usage
import pandas as pd

data = pd.read_csv('data/australian.dat', sep=' ', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

evaluator = Evaluator()
methods = ['pca', 'random_projection', 'kernel_pca', 'autoencoder', 'clustering', 'statistical_transform', 'feature_split']

for method in methods:
    print(f"\nMethod: {method}")
    results = evaluator.evaluate(X, y, test_size=0.3, random_state=42, label_fraction=0.1, method=method)
    print(f"Accuracy with transformed subspace: {results['accuracy_transformed']:.4f}")
    print(f"Accuracy with original features: {results['accuracy_original']:.4f}")
