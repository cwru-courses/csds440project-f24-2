import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
import scipy.sparse as sparse

# Class to perform the label propagation algorithm
class LabelPropagation:

    """
    Parameters:
    - sigma: Gaussian kernel width. Larger values mean dimilarity
           decays slower with dist bw points
    
    - max_iter: Maximum number of iterations for label propagation

    - tol: Convergence threshold. The algorithm will stop when the
         maximum change in labels between iters is less than this

    - alpha: Weight assigned to numerical similarity

    - beta: Weight assigned to categorical similarity
    """
    def __init__(self, sigma=1.0, max_iter=1000, tol=1e-3, alpha=0.5, beta=0.5, dynamic_weights=True):
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.beta = beta
        self.dynamic_weights = dynamic_weights

    """
    Fit the label propagation model

    Parameters:
    - X: Input data matrix

    - y_labeled: labels for labeled data

    - labeled_idx: indices of labeled points in X

    Returns an object that is the instance itself
    """
    def fit(self, X_num, X_cat, y_labeled, labeled_idx):
        print("Starting the fit process...")
        n_samples = X_num.shape[0]
        n_classes = len(np.unique(y_labeled))

        # Initialize the label matrix using one-hot encoding
        print("Initializing label matrix...")
        Y = np.zeros((n_samples, n_classes))
        Y[labeled_idx] = self._one_hot_encode(y_labeled)

        # Build the graph
        print("Building similarity graph...")
        W, D, L = self._build_graph(X_num, X_cat)
        print("Graph construction complete.")

        # Propagate labels
        if self.dynamic_weights:
            print("Starting dynamic label propagation...")
            self.labeled_distributions_ = self._propagate_labels_dynamic(W, D, L, Y, labeled_idx)
        else:
            print("Starting standard label propagation...")
            self.labeled_distributions_ = self._propagate_labels(L, Y, labeled_idx)
        print("Label propagation complete.")

        # Return the instance itself
        return self
        
    """
    Construct the similarity graph from the input data

    Parameters:
    - X: Input data matrix

    Returns
    - W: Similarity matrix where W[i,j] is similarity between points i and j

    - D: Diagonal degree matrix where D[i,i] is the sum of row i in W

    - P: Transition matrix P = D^-1 * W
    """
    def _build_graph(self, X_num, X_cat):
        # Compute similarity matrix using gaussian kernel
        print("Computing similarity matrix using RBF kernel...")
        W_num = rbf_kernel(X_num, X_num, gamma = 1 / (2 * self.sigma ** 2))
        W_cat = 1 - pairwise_distances(X_cat, metric='hamming')

        W_num = W_num / W_num.max()
        W_cat = W_cat / W_cat.max()

        W_combined = self.alpha * W_num + self.beta * W_cat
        # W = rbf_kernel(X, X, gamma = 1 / (2 * self.sigma ** 2))

        # Convert to kNN graph
        k = 10
        # For each row, zero out all but k largest values
        n_samples = X_num.shape[0]
        # Get indices of k largest values per row
        indices = np.argsort(W_combined, axis=1)[:, :-k]
        # Zero out elements not in kNN
        for i in range(n_samples):
            W_combined[i, indices[i]] = 0
        # Make symmetric
        W_combined = np.maximum(W_combined, W_combined.T)
        
        # Compute degree matrix
        D = np.diag(np.sum(W_combined, axis=1))
        
        # Compute graph Laplacian 
        L = D - W_combined
        print("Graph Laplacian constructed.")
        
        return W_combined, D, L
        
    """
    Run the label propagation algorithm

    Parameters:
    - P: Transition matrix for label propagation

    - y_initial: Initial labels

    - labeled_idx: Indices of labeled points
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
    Propagate labels with dynamic weight updates

    RESEARCH EXTENSION IMPLEMENTATIONS
    """
    def _propagate_labels_dynamic(self, W, D, L, Y, labeled_idx):
        n_samples = Y.shape[0]
        unlabeled_idx = np.setdiff1d(np.arange(n_samples), labeled_idx)
        
        # Initialize
        current_labels = Y.copy()
        prev_labels = np.zeros_like(Y)
        
        for iter in range(self.max_iter):
            # Update weights based on current predictions
            W = self._update_weights(W, current_labels, labeled_idx)
            
            # Update D and L with new weights
            D = np.diag(np.sum(W, axis=1))
            L = D - W
            
            # Partition updated Laplacian
            Luu = L[np.ix_(unlabeled_idx, unlabeled_idx)]
            Lul = L[np.ix_(unlabeled_idx, labeled_idx)]
            
            # Solve harmonic function with updated weights
            fu = -np.linalg.inv(Luu) @ Lul @ Y[labeled_idx]
            
            # Update labels
            current_labels[unlabeled_idx] = fu
            
            # Check convergence
            if np.abs(current_labels - prev_labels).max() < self.tol:
                print(f"Converged after {iter + 1} iterations")
                break
                
            prev_labels = current_labels.copy()
        
        return current_labels

    def _update_weights(self, W, current_labels, labeled_idx):
        n_samples = W.shape[0]
        current_pred = np.argmax(current_labels, axis=1)

        agreement_matrix = np.zeros_like(W)
        for i in range(n_samples):
            for j in range(n_samples):
                if W[i, j] > 0:
                    if current_pred[i] == current_pred[j]:
                        agreement_matrix[i,j] = 1.0
                    else:
                        agreement_matrix[i, j] = 0.5

        W_new = W * agreement_matrix

        W_new = W_new / (W_new.max() + 1e-10)

        W_new = np.maximum(W_new, W_new.T)

        return W_new

    """
    Convert the label vector to one-hot encoded matrix
    Helps to ensure all classes are treated equally, if data has more than 2 classes
    
    Parameters:
    - y: Vector of class labels
    
    Returns one_hot: One hot encoded matrix where each row corresponds to a label in y
    """
    def _one_hot_encode(self, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_samples = len(y)

        # Create matrix that is num_samples x num_classes
        one_hot = np.zeros((n_samples, n_classes))

        label_to_index = {label: idx for idx, label in enumerate(classes)}

        # Everything other than the corresponding class of the instance should be 0
        for i, label in enumerate(y):
            one_hot[i, label_to_index[label]] = 1

        # Return the one-hot encoded label matrix
        return one_hot

    """
    Parameters:
    - return_distributions: if true, return raw scores. If false, return binary predictions

    Returns predictions: 
        if return_distributions=False, return predicted class labels
        if return_distributions=True, returns probability distributions
    """
    def predict(self, return_distributions=False):
        if return_distributions:
            return self.labeled_distributions_
        else:
            return np.argmax(self.labeled_distributions_, axis=1)