import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def t_func(z):
    """
    Binary function t_{1, 2}(z) that returns a binary vector where the index of the 
    largest value in z is marked with 1, and others are 0. 
    If there are multiple maximum values - choose the smallest index is chosen.
    
    Parameters:
    z1 or z2 (numpy array): Input vector
    
    Returns:
    numpy array: Binary vector with 1 at the index of the maximum value.
    """
    # Find the index of the maximum value. In case of ties, np.argmax returns the smallest index.
    max_index = np.argmax(z)
    
    # Create a binary vector where only the max_index position is 1, and others are 0
    t_vector = np.zeros_like(z, dtype=int)
    t_vector[max_index] = 1
    
    return t_vector

class LinearSubspaceModel(BaseEstimator, TransformerMixin):
    """
    Linear Subspace Model (LS) for semi-supervised learning.
    """

    def __init__(self, m=10, p=5, ridge_alpha=1.0):
        """
        Parameters:
        - m: Number of binary functions tℓ for each view.
        - p: Number of singular vectors to keep.
        - ridge_alpha: Regularization parameter for ridge regression.
        """
        self.m = m  # Number of binary functions for each view
        self.p = p  # Number of singular vectors to keep
        self.ridge_alpha = ridge_alpha  # Regularization parameter for Ridge regression
        self.W1 = None  # Weight matrix for view 1
        self.W2 = None  # Weight matrix for view 2
        self.V1 = None  # Left singular vectors for view 1
        self.V2 = None  # Left singular vectors for view 2

    def compute_weight_matrix(self, Z1, Z2):
        """
        Compute the weight vectors by solving the optimization problem.

        Args:
            Z1: First view of the unlabeled data (features).
            Z2: Second view of the unlabeled data (features).

        Returns:
            W: Weight matrix where each column corresponds to a weight vector.
        """
        W = np.zeros((Z2.shape[1], self.m))  # Initialize weight matrix
        
        # Loop over each binary function
        for i in range(self.m):
            ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            
            # Create the target vector for this binary function t_l (for each i)
            t_l = np.array([t_func(z) for z in Z1])[:, i]  # Apply t_func to each row of Z1
            
            # Fit ridge regression for the binary function
            ridge.fit(Z2, t_l)  # Z2 is the features, and t_l is the target for this binary function
            
            # Store weight vector for this binary function
            W[:, i] = ridge.coef_
        
        return W

    def compute_singular_vectors(self, W):
        print("W:", W.shape)
        """
        Compute the top p left singular vectors of matrix W using SVD.

        Args:
            W: Weight matrix.

        Returns:
            V: The top p left singular vectors of W.
        """
        svd = TruncatedSVD(n_components=self.p)
        svd.fit(W.T)
        print("SVD:",svd.components_.shape)
        return svd.components_.T  # Return the top p left singular vectors

    def fit(self, Z1, Z2):
        """
        Train the Linear Subspace Model using only unlabeled data.

        Args:
            Z1: First view of the unlabeled data.
            Z2: Second view of the unlabeled data.
        """

        # Step 2: Compute the weight matrix W2 for the second view using T1 as binary functions
        self.W2 = self.compute_weight_matrix(Z1, Z2)
        self.V2 = self.compute_singular_vectors(self.W2)

        # Step 3: Compute the weight matrix W1 for the first view using T2_binary as binary functions
        self.W1 = self.compute_weight_matrix(Z2, Z1)
        self.V1 = self.compute_singular_vectors(self.W1)

        return self

    def transform(self, Z1, Z2):
        """
        Generate the new 2p-dimensional feature vector for each (z1, z2).

        Args:
            Z1: First view of the data.
            Z2: Second view of the data.

        Returns:
            Transformed feature vectors.
        """
        #print("Z1:", Z1.shape)
        #print("Z2:", Z2.shape)

        z1_transformed = Z1 @ self.V1  # Project Z1 onto the singular vectors of W1
        z2_transformed = Z2 @ self.V2  # Project Z2 onto the singular vectors of W2
        
        # Combine theoriginal features with the transformed ones
        # Experiment with performance when only using transformed...
        new_features =  np.hstack([Z1, Z2, z1_transformed, z2_transformed])

        #print(new_features.shape)
        return new_features
def semi_supervised_learning(X1_labeled, X2_labeled, y_labeled, X1_unlabeled, X2_unlabeled):
    """
    Perform semi-supervised learning using the Linear Subspace Model.
    
    Args:
        X1_labeled, X2_labeled: Labeled data for views 1 and 2.
        y_labeled: Labels for the labeled data.
        X1_unlabeled, X2_unlabeled: Unlabeled data for views 1 and 2.
    Returns:
        trained_classifier: The trained classifier.
    """
    # Step 1: Train the Linear Subspace Model
    model = LinearSubspaceModel(m=10, p=5, ridge_alpha=1.0)
    model.fit(X1_unlabeled, X2_unlabeled)  # Fit the model on the unlabeled data

    # Step 2: Transform both labeled and unlabeled data
    labeled_features = model.transform(X1_labeled, X2_labeled)
    unlabeled_features = model.transform(X1_unlabeled, X2_unlabeled)

    # Step 3: Train a classifier on the labeled data
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(labeled_features, y_labeled)

    # Step 4: Generate pseudo-labels for the unlabeled data
    pseudo_labels = classifier.predict(unlabeled_features)

    # Step 5: Combine labeled and pseudo-labeled data
    combined_features = np.vstack([labeled_features, unlabeled_features])
    combined_labels = np.hstack([y_labeled, pseudo_labels])

    # Step 6: Retrain the classifier with both labeled and pseudo-labeled data
    classifier.fit(combined_features, combined_labels)

    return classifier

# Dataset and Train-Test Split
digits = load_digits()
X = digits.data  
y = digits.target  

# Split into train/test set (30% test, 70% train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the two views: one from the original data and another from PCA
pca = PCA(n_components=20)  
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Use the original features for view 1 and PCA-transformed features for view 2
X1_train, X2_train = X_train, X_train_pca
X1_test, X2_test = X_test, X_test_pca

# --- Use only 10% of the labeled data for training the logistic regression classifier ---
X1_train_small, X1_val, y_train_small, y_val = train_test_split(X1_train, y_train, test_size=0.99, random_state=42)
X2_train_small, X2_val, _, _ = train_test_split(X2_train, y_train, test_size=0.99, random_state=42)

# --- Linear Subspace Logistic Regression ---
print("Training Linear Subspace Logistic Regression on full unlabeled data...")

# Train the Linear Subspace Model using the full unlabeled data
model = LinearSubspaceModel(m=10, p=5, ridge_alpha=1.0)
model.fit(X1_train, X2_train)  # Fit on full unlabeled data

# Transform the training, validation, and test sets using the learned subspace
X_train_transformed = model.transform(X1_train, X2_train)
X_train_small_transformed = model.transform(X1_train_small, X2_train_small)
X_val_transformed = model.transform(X1_val, X2_val)
X_test_transformed = model.transform(X1_test, X2_test)

# Train a logistic regression classifier on only 10% of the labeled data
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_small_transformed, y_train_small)  # Train on 10% labeled data

# Make predictions on the validation set
y_pred_val = clf.predict(X_val_transformed)

# Evaluate the accuracy of the model on the validation set
accuracy = accuracy_score(y_val, y_pred_val)
print(f"Accuracy on validation set with Linear Subspace Logistic Regression (10% labeled data): {accuracy:.4f}")

# --- Regular Logistic Regression on 10% labeled data ---
print("Training Regular Logistic Regression on 10% labeled data...")

# Train a regular logistic regression classifier using only the 10% labeled data
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X1_train_small, y_train_small)  # Train on 10% labeled data
y_pred_log_reg = log_reg.predict(X1_val)  # Predict on validation set

# Evaluate the accuracy of the regular logistic regression model
accuracy_log_reg = accuracy_score(y_val, y_pred_log_reg)
print(f"Accuracy of Regular Logistic Regression on 10% labeled data: {accuracy_log_reg:.4f}")
