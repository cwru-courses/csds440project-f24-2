import numpy as np
from harmonic_function import HarmonicFunction
from label_propagation import LabelPropagation
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_mnist_subset(n_samples=1000):
    print("Fetching MNIST data...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Convert labels to integers
    y = y.astype(int)
    
    # Take subset
    indices = np.random.choice(len(X), n_samples, replace=False)
    X = X[indices]
    y = y[indices]
    
    # Normalize pixel values to [0,1]
    X = X.astype(float) / 255.0
    
    return X, y

def evaluate_classification(y_true, y_pred):
    return np.mean(y_true == y_pred)

def apply_cmn(label_distributions, y_labeled):
    """Apply Class Mass Normalization as described in paper section 4"""
    n_samples = label_distributions.shape[0]
    n_classes = label_distributions.shape[1]
    
    # Calculate prior class probabilities from labeled data
    priors = np.zeros(n_classes)
    for i in range(n_classes):
        priors[i] = np.mean(y_labeled == i)
    
    # Calculate class masses
    masses = label_distributions.sum(axis=0)
    
    # Adjust distributions
    adjusted_distributions = np.zeros_like(label_distributions)
    for i in range(n_classes):
        adjusted_distributions[:, i] = label_distributions[:, i] * priors[i] / masses[i]
    
    # Normalize rows to sum to 1
    row_sums = adjusted_distributions.sum(axis=1)
    adjusted_distributions /= row_sums[:, np.newaxis]
    
    return adjusted_distributions

def main():
    # Parameters
    n_samples = 60000
    n_labeled = 6000
    sigma = 380.0  # Using paper's value for MNIST
    
    # Load data
    X, y = load_mnist_subset(n_samples)
    
    # Split data into labeled and unlabeled sets
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
        X, y, test_size=(n_samples-n_labeled)/n_samples, 
        random_state=42, stratify=y
    )
    
    # Create indices for labeled data
    labeled_idx = np.arange(n_labeled)
    
    # Combine data
    X_combined = np.vstack([X_labeled, X_unlabeled])
    y_combined = np.concatenate([y_labeled, y_unlabeled])
    
    print(f"Dataset shapes:")
    print(f"Total data: {X_combined.shape}")
    print(f"Labeled samples: {X_labeled.shape[0]}")
    print(f"Unlabeled samples: {X_unlabeled.shape[0]}")
    print(f"Number of classes: {len(np.unique(y_combined))}")
    
    # Print class distribution in labeled data
    print("\nClass distribution in labeled data:")
    for i in range(10):
        count = np.sum(y_labeled == i)
        print(f"Class {i}: {count} samples")
    
    print("\nRunning labelpropagation learning...")
    lp = LabelPropagation(sigma=sigma)
    lp.fit(X_combined, y_labeled, labeled_idx)
    
    predictions = lp.predict(return_distributions=False)
    
    # Initialize and run label propagation
    """print("\nRunning harmonic function learning...")
    hf = HarmonicFunction(sigma=sigma)
    hf.fit(X_combined, y_labeled, labeled_idx)
    
    # Apply CMN to get better class balance
    distributions = hf.predict(return_distributions=True)
    adjusted_distributions = apply_cmn(distributions, y_labeled)
    predictions = np.argmax(adjusted_distributions, axis=1)"""
    
    # Evaluate
    overall_accuracy = evaluate_classification(y_combined, predictions)
    labeled_accuracy = evaluate_classification(y_labeled, predictions[labeled_idx])
    unlabeled_accuracy = evaluate_classification(y_unlabeled, predictions[n_labeled:])
    
    print(f"\nResults:")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"Labeled data accuracy: {labeled_accuracy:.4f}")
    print(f"Unlabeled data accuracy: {unlabeled_accuracy:.4f}")

if __name__ == "__main__":
    main()