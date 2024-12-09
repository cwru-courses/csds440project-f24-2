import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# THIS HAS TO BE DONE WITH THE OTHER VERSION OF THE LINEAR SUBSPACE MODEL BECAUSE IT KINDA WEIRDLY HANDLES ONE HOT ENCODINGS

def generate_feature_sets(s, num_features_F1, num_features_F2):
    """
    Generate two sets of features F1 and F2.
    - F1: Features from the first view (e.g., pixel patches, color histograms).
    - F2: Features from the second view (e.g., CNN embeddings, other transformations).
    
    F1 has 'num_features_F1' features, F2 has 'num_features_F2' features.
    """
    # F1 and F2 are initially just lists of indices
    F1 = list(range(num_features_F1))
    F2 = list(range(num_features_F2))

    # Apply one-hot encoding to the feature sets F1 and F2
    encoder = OneHotEncoder(sparse_output=False)
    
    F1_encoded = encoder.fit_transform(np.array(F1).reshape(-1, 1))
    F2_encoded = encoder.fit_transform(np.array(F2).reshape(-1, 1))

    return F1_encoded, F2_encoded

def generate_class_feature_arrays(F1_encoded, F2_encoded, num_classes, t):
    """
    Generate class-specific feature arrays by randomly sampling from encoded F1 and F2.
    """
    class_features = {}
    for y in range(1, num_classes + 1):
        T1_y = random.sample(list(F1_encoded), t)  # Randomly pick 't' features from F1
        T2_y = random.sample(list(F2_encoded), t)  # Randomly pick 't' features from F2
        class_features[y] = (T1_y, T2_y)
    return class_features

def generate_data_point_with_dependency(class_features, q1, q2, r):
    """
    Generate a data point by selecting features from T1_y and T2_y with dependencies.
    The parameters q1 and q2 specify how many features to pick from T1_y and T2_y, respectively.
    r controls the degree of dependency between the two views.
    """
    y = random.choice(list(class_features.keys()))  # Randomly select a class
    T1_y, T2_y = class_features[y]  # Get the feature arrays for this class
    
    # Choose a starting index for view 1 (F1) and view 2 (F2)
    p1 = random.randint(0, len(T1_y) - 1)  # Random index for F1
    p2 = random.randint(0, len(T2_y) - 1)  # Random index for F2
    
    # Introduce dependency between the two views with probability r
    if random.random() < r:
        p2 = p1  # Set p2 = p1 to introduce dependency between the views
    
    # Draw consecutive tokens based on p1 and p2 for F1 and F2
    features_from_T1 = [T1_y[(p1 + i) % len(T1_y)] for i in range(q1)]
    features_from_T2 = [T2_y[(p2 + i) % len(T2_y)] for i in range(q2)]
    
    feature_vector_F1 = features_from_T1  # View 1 feature vector
    feature_vector_F2 = features_from_T2  # View 2 feature vector

    return feature_vector_F1, feature_vector_F2, y  # Return features from both views and the label

def generate_dataset(s, num_classes, t, q1, q2, num_samples, r):
    """
    Generate a dataset with one-hot encoded features and labels.
    Returns the data split into two views (F1 and F2), including the dependencies between views.
    """
    F1_encoded, F2_encoded = generate_feature_sets(s, s, s)
    class_features = generate_class_feature_arrays(F1_encoded, F2_encoded, num_classes, t)
    dataset_F1 = []  # Store features from view F1
    dataset_F2 = []  # Store features from view F2
    labels = []  # Store the labels

    for _ in range(num_samples):
        feature_vector_F1, feature_vector_F2, label = generate_data_point_with_dependency(class_features, q1, q2, r)
        dataset_F1.append(feature_vector_F1)
        dataset_F2.append(feature_vector_F2)
        labels.append(label)

    return np.array(dataset_F1), np.array(dataset_F2), np.array(labels)

# Parameters for dataset generation
s = 20  # Size of vocabulary
num_classes = 4  # Number of classes (K)
t = 10  # Size of class vocabulary
q1 = 6  # Features to pick from T1_y
q2 = 6  # Features to pick from T2_y
num_samples = 100  # Number of data points to generate
r = 0.3  # Probability of introducing dependency between the two views

# Generate the dataset, split into views F1 and F2 with dependencies
#dataset_F1, dataset_F2, labels = generate_dataset(s, num_classes, t, q1, q2, num_samples, r)