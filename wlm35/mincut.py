import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
import networkx as nx

class MinCut:
    """
    Parameters:
    - sigma: Gaussian kernel width for similarity computation
    
    - alpha: Weight for numeric similarity
        
    - beta: Weight for categorical similarity
    """
    def __init__(self, sigma=1.0, alpha=0.5, beta=0.5, balance_factor=1.0):
        print("This includes the updates2")
        self.sigma = sigma
        self.large_weight = 1e6
        self.alpha = alpha
        self.beta = beta
        self.balance_factor = balance_factor

    """
    Fit the model 
    
    Parameters:
    - X_num: Numeric features
        
    - X_cat: Categorical features
        
    - y_labeled: Labels for labeled data
        
    - labeled_idx: Indices of labeled points in X

    Returns an instance which represents self fitted to the data
    """
    def fit(self, X_num, X_cat, y_labeled, labeled_idx):
        
        n_samples = X_num.shape[0]
    
        # Get positive and negative label indices
        pos_idx = labeled_idx[y_labeled == 1]
        neg_idx = labeled_idx[y_labeled == 0]
        
        # Build basic similarity graph
        G = nx.DiGraph()
        
        # Add example vertices
        for i in range(n_samples):
            G.add_node(i)
            
        # Add classification vertices v+ and v-
        G.add_node('v+') 
        G.add_node('v-')
        
        # Calculate class proportions from labeled data
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        expected_pos_ratio = n_pos / (n_pos + n_neg)
        
        # Connect labeled examples with scaled weights
        for idx in pos_idx:
            G.add_edge('v+', idx, capacity=self.large_weight * (1/n_pos))
        for idx in neg_idx:
            G.add_edge(idx, 'v-', capacity=self.large_weight * (1/n_neg))
        
        # Add weighted edges between examples with balancing term
        W = self._compute_similarity(X_num, X_cat)
        balance_term = expected_pos_ratio * (1 - expected_pos_ratio)
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if W[i,j] > 0:
                    # Scale weight by balance term
                    weight = W[i,j] * balance_term
                    G.add_edge(i, j, capacity=weight)
                    G.add_edge(j, i, capacity=weight)
        
        # Add weak edges from unlabeled points to v+ and v- to encourage balance
        unlabeled_idx = np.setdiff1d(range(n_samples), labeled_idx)
        unlabeled_weight = 0.1  # Small weight for unlabeled connections
        
        for idx in unlabeled_idx:
            G.add_edge('v+', idx, capacity=unlabeled_weight * expected_pos_ratio)
            G.add_edge(idx, 'v-', capacity=unlabeled_weight * (1 - expected_pos_ratio))

        try:
            cut_value, partition = nx.minimum_cut(G, 'v+', 'v-', capacity='capacity')
            
            # Convert partition to labels
            self.labels_ = np.zeros(n_samples)
            reachable, non_reachable = partition
            for node in reachable:
                if isinstance(node, int):
                    self.labels_[node] = 1
                    
            print(f"Cut value: {cut_value}")
            print(f"Partition sizes: {len(reachable)}, {len(non_reachable)}")
            print(f"Positive predictions: {np.sum(self.labels_)}")
            print(f"Negative predictions: {len(self.labels_) - np.sum(self.labels_)}")
            
        except nx.NetworkXError as e:
            print(f"Error in minimum cut: {e}")
            raise
                
        return self

    def _compute_similarity(self, X_num, X_cat):
        """Compute similarity matrix using distance metric from paper"""
        if X_num.shape[1] > 0:
            W_num = rbf_kernel(X_num, X_num, gamma=1/(2 * self.sigma**2))
            W_num = W_num / W_num.max()  # Normalize
        else:
            W_num = 0
            
        if X_cat.shape[1] > 0:
            W_cat = 1 - pairwise_distances(X_cat, metric='hamming')
            W_cat = W_cat / W_cat.max()  # Normalize
        else:
            W_cat = 0
            
        W = (W_num + W_cat) / 2
        return W

    """
    Construct similarity graph

    Parameters:
    - X_num: Numeric features
        
    - X_cat: Categorical features
    """
    def _build_graph(self, X_num, X_cat):
        
        # Compute similarity matrices
        W_num = rbf_kernel(X_num, X_num, gamma=1/(2 * self.sigma**2))
        W_cat = 1 - pairwise_distances(X_cat, metric='hamming')
        
        # Combine similarities
        W_num = W_num / W_num.max()
        W_cat = W_cat / W_cat.max()
        W = self.alpha * W_num + self.beta * W_cat
        
        # Create networkx graph
        G = nx.Graph()

        for i in range(self.n_samples):
            G.add_node(i)
        
        # Add nodes and edges
        node_degrees = np.sum(W, axis=1)
        for i in range(self.n_samples):
            for j in range(i+1, self.n_samples):
                if W[i,j] > 0:
                    normalized_weight = W[i,j] / np.sqrt(node_degrees[i] * node_degrees[j])
                    G.add_edge(i, j, weight=normalized_weight)
        
        return G

    """
    Add source (s) and sink (t) nodes and connect to labeled examples

    Parameters:
    - G: The graph

    - pos_idx: the indices of the positively labeled samples

    - neg_idx: the indices of the negatively labeled samples
    
    """
    def _add_terminal_nodes(self, G, pos_idx, neg_idx):
        
        # Add terminal nodes
        G.add_node('s')  # source
        G.add_node('t')  # sink

        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        total = n_pos + n_neg

        pos_weight = self.balance_factor * (total / (2 * n_pos)) if n_pos > 0 else np.inf
        neg_weight = self.balance_factor * (total / (2 * n_neg)) if n_neg > 0 else np.inf
        
        # Connect source to positive examples
        for idx in pos_idx:
            G.add_edge('s', idx, weight=pos_weight)
            
        # Connect negative examples to sink
        for idx in neg_idx:
            G.add_edge(idx, 't', weight=neg_weight)

    def predict(self):
        
        return self.labels_