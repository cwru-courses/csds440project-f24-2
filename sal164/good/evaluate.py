import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from linear_subspace_final import LinearSubspaceModel

def cross_validate_linear_subspace(
    dataset_F1, 
    dataset_F2, 
    labels, 
    unlabeled_proportion=0.05, 
    n_splits=5, 
    random_state=42,
    t_func="t_func_original",
    concatenate=True,
    p = 10,
    m = 5
):
    """Performs 5-fold cross-validation for both the transformed and baseline models.

    Args:
        dataset_F1 (array-like): Feature set 1.
        dataset_F2 (array-like): Feature set 2.
        labels (array-like): Target labels.
        unlabeled_proportion (float): Proportion of training data to use as unlabeled.
        n_splits (int): Number of cross-validation folds.
        random_state (int): Random state for reproducibility.

    Returns:
        dict: A dictionary containing evaluation metrics for both the transformed and baseline models.
    """
    results = {
        'transformed': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'tpr': [], 'fpr': []},
        'baseline': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'tpr': [], 'fpr': []}
    }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_idx, (train_index, test_index) in enumerate(kf.split(dataset_F1)):
        print(f"\n--- Fold {fold_idx + 1} ---")
        
        # Split into training and testing sets for this fold
        X_train_F1, X_test_F1 = dataset_F1[train_index], dataset_F1[test_index]
        X_train_F2, X_test_F2 = dataset_F2[train_index], dataset_F2[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Further split training data to get labeled and unlabeled sets
        X1_train_unlabeled, X1_train_labeled, X2_train_unlabeled, X2_train_labeled, y_train_unlabeled, y_train_labeled = train_test_split(
            X_train_F1, X_train_F2, y_train, test_size=1 - unlabeled_proportion, random_state=random_state
        )
        print("X1_train_unlabeled", X1_train_unlabeled.shape)
        print("X2_train_unlabeled", X2_train_unlabeled.shape)

        # --- Train and Evaluate Transformed Model ---
        model = LinearSubspaceModel(m=m, p = p, concatenate=concatenate, t_function_name=t_func)
        model.fit(X1_train_unlabeled, X2_train_unlabeled)  # Train model on both views

        X_train_transformed = model.transform(X1_train_labeled, X2_train_labeled)  # Transform training data
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_transformed, y_train_labeled)  # Train logistic regression on transformed features

        # Test set transformation and prediction
        X_test_transformed = model.transform(X_test_F1, X_test_F2)
        y_pred_transformed = clf.predict(X_test_transformed)

        # Calculate performance metrics for the transformed model
        accuracy = accuracy_score(y_test, y_pred_transformed)
        precision = precision_score(y_test, y_pred_transformed, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred_transformed, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred_transformed, average='weighted', zero_division=1)
        
        cm1 = confusion_matrix(y_test, y_pred_transformed)

        # Check if the problem is binary
        if cm1.shape == (2, 2):  # Binary classification only
            tn, fp, fn, tp = cm1.ravel()  # Unpack the 2x2 confusion matrix
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0  # True Positive Rate
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Positive Rate
        else:
            tpr = None  # Not defined for multiclass
            fpr = None  # Not defined for multiclass
        
        # Store results for this fold
        results['transformed']['accuracy'].append(accuracy)
        results['transformed']['precision'].append(precision)
        results['transformed']['recall'].append(recall)
        results['transformed']['f1'].append(f1)
        results['transformed']['tpr'].append(tpr)
        results['transformed']['fpr'].append(fpr)

        print(f"Transformed Model - Fold {fold_idx + 1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        # --- Train and Evaluate Baseline Model ---
        X_train_overall = np.concatenate([X1_train_labeled, X2_train_labeled], axis=1)  # Concatenate features
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_overall, y_train_labeled)  # Train logistic regression on combined features

        X_test_overall = np.concatenate([X_test_F1, X_test_F2], axis=1)  # Concatenate test features
        y_pred_baseline = clf.predict(X_test_overall)  # Predict using the baseline model

        # Calculate performance metrics for the baseline model
        base_accuracy = accuracy_score(y_test, y_pred_baseline)
        base_precision = precision_score(y_test, y_pred_baseline, average='weighted', zero_division=1)
        base_recall = recall_score(y_test, y_pred_baseline, average='weighted', zero_division=1)
        base_f1 = f1_score(y_test, y_pred_baseline, average='weighted', zero_division=1)
        
        cm = confusion_matrix(y_test, y_pred_baseline)
        print(cm)

        # Check if the problem is binary
        if cm.shape == (2, 2):  # Binary classification only
            tn, fp, fn, tp = cm.ravel()  # Unpack the 2x2 confusion matrix
            base_tpr = tp / (tp + fn) if (tp + fn) != 0 else 0  # True Positive Rate
            base_fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Positive Rate
        else:
            base_tpr = None  # Not defined for multiclass
            base_fpr = None  # Not defined for multiclass

        # Store results for this fold
        results['baseline']['accuracy'].append(base_accuracy)
        results['baseline']['precision'].append(base_precision)
        results['baseline']['recall'].append(base_recall)
        results['baseline']['f1'].append(base_f1)
        results['baseline']['tpr'].append(base_tpr)
        results['baseline']['fpr'].append(base_fpr)

        print(f"Baseline Model - Fold {fold_idx + 1}: Accuracy={base_accuracy:.4f}, Precision={base_precision:.4f}, Recall={base_recall:.4f}, F1={base_f1:.4f}")

    return results
