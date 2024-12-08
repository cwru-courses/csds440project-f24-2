import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, ParameterGrid
from sklearn.preprocessing import LabelEncoder, StandardScaler
from label_propagation import LabelPropagation
import matplotlib.pyplot as plt

def load_data(filepath):
    try:
        # Try reading with space delimiter first
        data = pd.read_csv(filepath, header=None, delimiter=r'\s+')
        
        if data.shape[1] == 1:  # If only one column, try comma delimiter
            data = pd.read_csv(filepath, header=None)
        
        # Separate features and target
        X = data.iloc[:, :-1]  # all columns except last
        y = data.iloc[:, -1]   # last column is target
        
        # Convert categorical labels to numeric starting from 0
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        
        # Handle numeric features
        X_num = X.copy()
        if len(categorical_cols) > 0:
            X_num[categorical_cols] = X_num[categorical_cols].apply(LabelEncoder().fit_transform)
        X_num = X_num.values.astype(float)
        
        # Normalize numeric features
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
        
        # Handle categorical features
        X_cat = X.copy()
        for col in X_cat.columns:
            if col in categorical_cols:
                X_cat[col] = LabelEncoder().fit_transform(X_cat[col].astype(str))
            else:
                # Bin numeric columns for categorical representation
                X_cat[col] = pd.qcut(X_cat[col], q=10, labels=False, duplicates='drop')
        X_cat = X_cat.values
        
        return X_num, X_cat, y
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def evaluate_hyperparameters(X_num, X_cat, y, labeled_idx, hyperparameters):
    results = []
    
    for params in ParameterGrid(hyperparameters):
        lp = LabelPropagation(**params)
        lp.fit(X_num, X_cat, y[labeled_idx], labeled_idx)
        predictions = lp.predict()
        
        unlabeled_idx = np.setdiff1d(np.arange(len(y)), labeled_idx)
        accuracy = np.mean(predictions[unlabeled_idx] == y[unlabeled_idx])
        
        results.append({**params, 'accuracy': accuracy})
    
    return pd.DataFrame(results)

def evaluate_split_sizes(X_num, X_cat, y, split_sizes, best_params):
    results = []
    
    for split_size in split_sizes:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=split_size, random_state=42)
        labeled_idx, _ = next(sss.split(X_num, y))
        y_labeled = y[labeled_idx]
        
        lp = LabelPropagation(**best_params)
        lp.fit(X_num, X_cat, y_labeled, labeled_idx)
        predictions = lp.predict()
        
        unlabeled_idx = np.setdiff1d(np.arange(len(y)), labeled_idx)
        accuracy = np.mean(predictions[unlabeled_idx] == y[unlabeled_idx])
        
        results.append({'split_size': split_size, 'accuracy': accuracy})
    
    return pd.DataFrame(results)

"""def main():
    # Load the data
    X_num, X_cat, y = load_data("wlm35/data/australian.dat")
    #X_num, X_cat, y = load_data("wlm35/data/kr-vs-kp.data")
    #X_num, X_cat, y = load_data("wlm35/data/hypothyroid.data")
    
    # Print some information about the data
    print(f"Number of unique classes: {len(np.unique(y))}")
    print(f"Label range: {y.min()} to {y.max()}")
    
    # Use stratified sampling for labeled data
    n_samples = len(y)
    labeled_size = int(0.1 * n_samples)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.7, random_state=42)
    labeled_idx, unlabeled_idx = next(sss.split(X_num, y))
    y_labeled = y[labeled_idx]
    
    print("Initializing Label Propagation model...")
    lp = LabelPropagation(
        sigma=0.5,
        max_iter=1000,
        tol=1e-3,
        alpha=0.99,
        beta=0.01,
        dynamic_weights=True
    )
    
    # Fit the model
    lp.fit(X_num, X_cat, y_labeled, labeled_idx)
    
    # Get predictions
    predictions = lp.predict()
    
    # Calculate accuracy on unlabeled data
    accuracy = np.mean(predictions[unlabeled_idx] == y[unlabeled_idx])
    
    # Calculate per-class accuracy
    print("\nPer-class performance on unlabeled data:")
    for c in np.unique(y):
        mask = (y[unlabeled_idx] == c)
        if sum(mask) > 0:
            class_acc = np.mean(predictions[unlabeled_idx][mask] == y[unlabeled_idx][mask])
            print(f"Class {c} accuracy: {class_acc:.4f} (n={sum(mask)})")
    
    print(f"\nOverall Results:")
    print(f"Number of labeled samples: {len(labeled_idx)}")
    print(f"Number of unlabeled samples: {len(unlabeled_idx)}")
    print(f"Overall accuracy on unlabeled data: {accuracy:.4f}")"""

def main_australian_dataset():
    # Load the data
    X_num, X_cat, y = load_data("wlm35/data/australian.dat")

    # Use stratified sampling for labeled data
    n_samples = len(y)
    labeled_size = int(0.1 * n_samples)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.9, random_state=42)
    labeled_idx, _ = next(sss.split(X_num, y))
    y_labeled = y[labeled_idx]

    # Define the hyperparameter grid
    hyperparameters = {
        'sigma': [0.01, 0.1, 0.5, 1.0, 10, 25, 50, 100],
        'max_iter': [500, 1000],
        'tol': [1e-3, 1e-4],
        'alpha': [.9, .99],
        'beta': [.1, 0.01],
        'dynamic_weights': [False]
    }

    # Perform hyperparameter tuning
    results_df = evaluate_hyperparameters(X_num, X_cat, y, labeled_idx, hyperparameters)

    # Find the best hyperparameters
    best_params = {k: v for k, v in results_df.loc[results_df['accuracy'].idxmax()].to_dict().items() if k != 'accuracy'}

    print("Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # Evaluate different split sizes
    split_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    split_results_df = evaluate_split_sizes(X_num, X_cat, y, split_sizes, best_params)

    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    results_df.groupby('sigma')['accuracy'].mean().plot(kind='line', marker='o', ax=ax)
    ax.set_title("Accuracy vs Sigma with Dynamic Weight Updates")
    ax.set_xlabel("Sigma")
    ax.set_ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot accuracy for different split sizes
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    split_results_df.plot(x='split_size', y='accuracy', kind='line', marker='o', ax=ax)
    ax.set_title("Accuracy vs Split Size with Dynamic Weight Updates")
    ax.set_xlabel("Test Size")
    ax.set_ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main_chess_dataset():
    # Load the data
    X_num, X_cat, y = load_data("wlm35/data/kr-vs-kp.data")

    # Use stratified sampling for labeled data
    n_samples = len(y)
    labeled_size = int(0.1 * n_samples)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.9, random_state=42)
    labeled_idx, _ = next(sss.split(X_num, y))
    y_labeled = y[labeled_idx]

    # Define the hyperparameter grid
    hyperparameters = {
        'sigma': [10],
        'max_iter': [500, 1000],
        'tol': [1e-3, 1e-4],
        'alpha': [.01, .1],
        'beta': [.99, 0.9],
        'dynamic_weights': [False]
    }

    # Perform hyperparameter tuning
    results_df = evaluate_hyperparameters(X_num, X_cat, y, labeled_idx, hyperparameters)

    # Find the best hyperparameters
    best_params = {k: v for k, v in results_df.loc[results_df['accuracy'].idxmax()].to_dict().items() if k != 'accuracy'}

    print("Best Hyperparameters for Chess Dataset:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # Evaluate different split sizes
    split_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    split_results_df = evaluate_split_sizes(X_num, X_cat, y, split_sizes, best_params)

    # Plot accuracy for different split sizes
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    split_results_df.plot(x='split_size', y='accuracy', kind='line', marker='o', ax=ax)
    ax.set_title("Accuracy vs Split Size with Dynamic Weights (Chess Dataset)")
    ax.set_xlabel("Test Size")
    ax.set_ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main_mnist_dataset():
    # Load MNIST dataset
    from sklearn.datasets import fetch_openml
    from sklearn.utils import resample
    
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Convert data to float32 for memory efficiency
    X = X.astype('float32')
    
    # Normalize pixel values
    X = X / 255.0
    
    # Create balanced sample of 6000 points (600 per class)
    samples_per_class = 600
    X_sampled = []
    y_sampled = []
    
    for class_label in range(10):
        class_mask = (y.astype(int) == class_label)
        X_class = X[class_mask]
        y_class = y[class_mask]
        
        # Random sample for this class
        if len(X_class) > samples_per_class:
            X_class_sample, y_class_sample = resample(
                X_class, 
                y_class,
                n_samples=samples_per_class,
                random_state=42 + class_label  # Different seed for each class
            )
            X_sampled.append(X_class_sample)
            y_sampled.append(y_class_sample)
    
    # Combine all sampled data
    X_final = np.vstack(X_sampled)
    y_final = np.concatenate(y_sampled)
    
    # Convert to numeric and categorical representations
    X_num = X_final  # Already numeric
    X_cat = (X_final > 0.5).astype(int)  # Binary categorical representation
    
    # Use stratified sampling for labeled data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.9, random_state=42)
    labeled_idx, _ = next(sss.split(X_num, y_final))
    
    # Define hyperparameter grid appropriate for MNIST
    hyperparameters = {
        'sigma': [0.1, 1.0, 10.0, 100.0, 500.0],  # Smaller range due to normalized pixel values
        'max_iter': [500],
        'tol': [1e-3],
        'alpha': [0.99],
        'beta': [0.01],
        'dynamic_weights': [True]
    }
    
    results_df = evaluate_hyperparameters(X_num, X_cat, y_final.astype(int), labeled_idx, hyperparameters)
    
    # Find best hyperparameters
    best_params = {k: v for k, v in results_df.loc[results_df['accuracy'].idxmax()].to_dict().items() 
                  if k != 'accuracy'}
    
    print("\nBest Hyperparameters for MNIST Dataset:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Evaluate different split sizes
    split_sizes = [0.7, 0.8, 0.9]  # Reduced split sizes due to computational intensity
    split_results_df = evaluate_split_sizes(X_num, X_cat, y_final.astype(int), 
                                          split_sizes, best_params)
    
    # Plot accuracy vs sigma
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    results_df.groupby('sigma')['accuracy'].mean().plot(kind='line', marker='o', ax=ax)
    ax.set_title("MNIST: Accuracy vs Sigma with Dynamic Weight Updates")
    ax.set_xlabel("Sigma")
    ax.set_ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot accuracy vs split size
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    split_results_df.plot(x='split_size', y='accuracy', kind='line', marker='o', ax=ax)
    ax.set_title("MNIST: Accuracy vs Split Size with Dynamic Weight Updates")
    ax.set_xlabel("Test Size")
    ax.set_ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_mnist_dataset()