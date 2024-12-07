import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import scipy.sparse as sparse

# Class to perform the label propagation algorithm
class LabelPropagation:

    """
    sigma: Gaussian kernel width. Larger values mean dimilarity
           decays slower with dist bw points
    
    max_iter: Maximum number of iterations for label propagation

    tol: Convergence threshold. The algorithm will stop when the
         maximum change in labels between iters is less than this
    """
    def __init__(self, sigma=1.0, max_iter=1000, tol=1e-3):
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol

    """
    Fit the label propagation model

    Parameters
    ----------
    X: Input data matrix

    y_labeled: labels for labeled data

    labeled_idx: indices of labeled points in X
    ----------

    Returns an object that is the instance itself
    """
    def fit(self, X, y_labeled, labeled_idx):
        print("Starting the fit process...")
        n_samples = X.shape[0]
        n_classes = len(np.unique(y_labeled))

        # Initialize the label matrix using one-hot encoding
        print("Initializing label matrix...")
        Y = np.zeros((n_samples, n_classes))
        Y[labeled_idx] = self._one_hot_encode(y_labeled)

        # Build the graph
        print("Building similarity graph...")
        W, D, P = self._build_graph(X)
        print("Graph construction complete.")

        # Propagate labels
        print("Starting label propagation...")
        self.labeled_distributions_ = self._propagate_labels(P, Y, labeled_idx)
        print("Label propagation complete.")

        # Return the instance itself
        return self
        
    """
    Construct the similarity graph from the input data

    Parameters
    ----------
    X: Input data matrix
    ----------

    Returns
    ----------
    W: Similarity matrix where W[i,j] is similarity between points i and j

    D: Diagonal degree matrix where D[i,i] is the sum of row i in W

    P: Transition matrix P = D^-1 * W
    ----------
    """
    def _build_graph(self, X):
        # Compute similarity matrix using gaussian kernel
        print("Computing similarity matrix using RBF kernel...")
        W = rbf_kernel(X, X, gamma = 1 / (2 * self.sigma ** 2))

        # Convert to kNN graph
        k = 10
        # For each row, zero out all but k largest values
        n_samples = X.shape[0]
        # Get indices of k largest values per row
        indices = np.argsort(W, axis=1)[:, :-k]
        # Zero out elements not in kNN
        for i in range(n_samples):
            W[i, indices[i]] = 0
        # Make symmetric
        W = np.maximum(W, W.T)
        
        # Compute degree matrix
        D = np.diag(np.sum(W, axis=1))
        
        # Compute graph Laplacian 
        L = D - W
        print("Graph Laplacian constructed.")
        
        return W, D, L
        
    """
    Run the label propagation algorithm

    Parameters
    ----------
    P: Transition matrix for label propagation

    y_initial: Initial labels

    labeled_idx: Indices of labeled points
    """
    def _propagate_labels(self, L, y_initial, labeled_idx):
        n_samples = y_initial.shape[0]
        # Partition Laplacian into labeled/unlabeled
        unlabeled_idx = np.setdiff1d(np.arange(n_samples), labeled_idx)
        Luu = L[np.ix_(unlabeled_idx, unlabeled_idx)]
        Lul = L[np.ix_(unlabeled_idx, labeled_idx)]
        
        # Solve harmonic function (from paper equation 5)
        fu = -np.linalg.inv(Luu) @ Lul @ y_initial[labeled_idx]
        
        # Combine labeled and unlabeled predictions
        f = y_initial.copy()
        f[unlabeled_idx] = fu
        return f

    """
    Convert the label vector to one-hot encoded matrix
    Helps to ensure all classes are treated equally, if data has more than 2 classes
    
    Parameters
    ----------
    y: Vector of class labels
    ----------
    
    Returns
    ----------
    one_hot: One hot encoded matrix where each row corresponds to a label in y
    ----------
    """
    def _one_hot_encode(self, y):
        n_classes = len(np.unique(y))
        n_samples = len(y)

        # Create matrix that is num_samples x num_classes
        one_hot = np.zeros((n_samples, n_classes))

        # Everything other than the corresponding class of the instance should be 0
        one_hot[np.arange(n_samples), y] = 1

        # Return the one-hot encoded label matrix
        return one_hot

    """
    Parameters
    ----------
    return_distributions: if true, return raw scores. If false, return binary predictions
    ----------

    Returns
    ----------
    predictions: 
        if return_distributions=False, return predicted class labels
        if return_distributions=True, returns probability distributions
    ----------
    """
    def predict(self, return_distributions=False):
        if return_distributions:
            return self.labeled_distributions_
        else:
            return np.argmax(self.labeled_distributions_, axis=1)