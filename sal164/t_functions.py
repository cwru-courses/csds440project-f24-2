import numpy as np

def t_func_original(self, z):
    max_index = np.argmax(z)  # Index of maximum value in z
    t_vector = np.zeros_like(z, dtype=int)  # Create zero vector of the same shape as z
    t_vector[max_index] = 1  # Set 1 at the position of the max value
    return t_vector


def t_func_softmax(self, z):
    exp_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
    softmax_vector = exp_z / np.sum(exp_z)
    return softmax_vector

def t_func_threshold(self, z, threshold=0.5):
    threshold_vector = (z > threshold).astype(int)
    return threshold_vector


def t_func_l2_normalized(self, z):
    norm = np.linalg.norm(z)
    normalized_vector = z / norm if norm > 0 else z  # Avoid division by zero
    return normalized_vector


def t_func_rank(self, z):
    sorted_indices = np.argsort(z)
    rank_vector = np.zeros_like(z)
    rank_vector[sorted_indices] = np.arange(len(z))
    return rank_vector

def t_func_top_k(self, z, k=3):
    top_k_indices = np.argsort(z)[-k:]  # Get the indices of the top k elements
    top_k_vector = np.zeros_like(z, dtype=int)
    top_k_vector[top_k_indices] = 1
    return top_k_vector

def t_func_binning(self, z, num_bins=5):
    min_z, max_z = np.min(z), np.max(z)
    bin_edges = np.linspace(min_z, max_z, num_bins + 1)
    bin_vector = np.digitize(z, bin_edges) - 1  # Subtract 1 for zero-based index
    return bin_vector

def t_func_zscore(self, z):
    mean = np.mean(z)
    std_dev = np.std(z)
    zscore_vector = (z - mean) / std_dev if std_dev > 0 else z
    return zscore_vector
