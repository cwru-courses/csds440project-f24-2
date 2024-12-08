import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
import t_functions

# Linear Subspace Model (LSM) and training code
class LinearSubspaceModel:
    def __init__(self, m=10, p=5, ridge_alpha=1.0, concatenate=True, t_function_name='t_func_original'):
        """
        Initialize the Linear Subspace Model.
        
        Parameters:
        - m: Number of basis vectors in the weight matrix.
        - p: Number of singular vectors to keep for dimensionality reduction.
        - ridge_alpha: Regularization strength for Ridge regression.
        """
        self.m = m  # Number of subspace components
        self.p = p  # Number of singular vectors to retain
        self.ridge_alpha = ridge_alpha  # Regularization parameter for Ridge regression
        self.W1 = None  # Weight matrix for Z2 -> Z1
        self.W2 = None  # Weight matrix for Z1 -> Z2
        self.V1 = None  # Singular vectors for Z2 -> Z1
        self.V2 = None  # Singular vectors for Z1 -> Z2
        self.concatenate = concatenate

        self.t_func = getattr(t_functions, t_function_name)

    def t_func_original(self, z):
        """
        One-hot encodes the index of the maximum value in the vector z.
        Used by original paper, but not well motivated.
        Used to parameterize the representation of P(y|zi).
        
        Parameters:
        - z: Input vector.
        
        Returns:
        - t_vector: One-hot encoded vector with 1 at the position of the max value in z.
        """
        max_index = np.argmax(z)  # Index of maximum value in z
        t_vector = np.zeros_like(z, dtype=int)  # Create zero vector of the same shape as z
        t_vector[max_index] = 1  # Set 1 at the position of the max value
        return t_vector

    def compute_weight_matrix(self, Z1, Z2):
        """
        Compute the weight matrix that maps Z2 to Z1.
        
        Parameters:
        - Z1: First input data matrix of shape (n_samples, n_features).
        - Z2: Second input data matrix of shape (n_samples, n_features).
        
        Returns:
        - W: Weight matrix of shape (n_features, m) where m is the number of basis vectors.
        """
        W = np.zeros((Z1.shape[1], self.m))  # Initialize weight matrix
        #print(Z1)
        for i in range(self.m):
            # Train a Ridge regression model to predict the i-th column of one-hot encodings of Z1
            ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            t_l = np.array([self.t_func(self, z) for z in Z1])[:, i]  # Extract the i-th column of one-hot encodings
            ridge.fit(Z2, t_l)  # Fit the Ridge regression model using Z2 as predictors
            W[:, i] = ridge.coef_  # Store the learned coefficients for the i-th column
        return W

    def compute_singular_vectors(self, W):
        """
        Compute the top p singular vectors of the transposed weight matrix W.
        
        Parameters:
        - W: Weight matrix of shape (n_features, m).
        
        Returns:
        - Singular vectors (V) of shape (n_features, p).
        """
        svd = TruncatedSVD(n_components=self.p)  # Truncated SVD for dimensionality reduction
        svd.fit(W.T)  # Fit SVD on the transpose of the weight matrix
        return svd.components_.T  # Return the singular vectors (transpose to match input shape)

    def fit(self, Z1, Z2):
        """
        Train the model using input data Z1 and Z2.
        
        Parameters:
        - Z1: First input data matrix of shape (n_samples, n_features).
        - Z2: Second input data matrix of shape (n_samples, n_features).
        """
        self.W2 = self.compute_weight_matrix(Z1, Z2)  # Compute weight matrix for Z1 -> Z2
        self.V2 = self.compute_singular_vectors(self.W2)  # Compute top p singular vectors for W2
        self.W1 = self.compute_weight_matrix(Z2, Z1)  # Compute weight matrix for Z2 -> Z1
        self.V1 = self.compute_singular_vectors(self.W1)  # Compute top p singular vectors for W1

    def transform(self, Z1, Z2):
        """
        Transform input data Z1 and Z2 using the learned subspace projections.
        
        Parameters:
        - Z1: First input data matrix of shape (n_samples, n_features).
        - Z2: Second input data matrix of shape (n_samples, n_features).
        
        Returns:
        - new_features: Concatenated feature representation of shape (n_samples, 2*n_features + 2*p).
          This includes the original features from Z1 and Z2 and the subspace-projected features.
        """
        z1_transformed = Z1 @ self.V1  # Project Z1 into the subspace using V1
        z2_transformed = Z2 @ self.V2  # Project Z2 into the subspace using V2
        # Concatenate original and projected features
        # But also test without doing this
        if self.concatenate:
            new_features = np.hstack([Z1, Z2, z1_transformed, z2_transformed])  
        else: 
            new_features = np.hstack([z1_transformed, z2_transformed])  
        return new_features
