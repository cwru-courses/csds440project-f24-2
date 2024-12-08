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
    def __init__(self, sigma=1.0, alpha=0.5, beta=0.5):
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

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
        
        print("Starting mincut fit process...")
        self.n_samples = X_num.shape[0]
        
        # Get positive and negative label indices
        pos_idx = labeled_idx[y_labeled == 1]
        neg_idx = labeled_idx[y_labeled == 0]
        
        # Build the graph
        print("Building similarity graph...")
        G = self._build_graph(X_num, X_cat)
        
        # Add source and sink nodes
        print("Adding source and sink nodes...")
        self._add_terminal_nodes(G, pos_idx, neg_idx)
        
        # Find the minimum cut
        print("Computing minimum cut...")
        cut_value, partition = nx.minimum_cut(G, 's', 't', capacity='weight')
        reachable, non_reachable = partition
        
        # Convert partition to labels
        self.labels_ = np.zeros(self.n_samples)
        for node in reachable:
            if isinstance(node, int):  # Skip source/sink nodes
                self.labels_[node] = 1
                
        print("Mincut completed.")
        return self

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
        
        # Add nodes and edges
        for i in range(self.n_samples):
            G.add_node(i)
            for j in range(i+1, self.n_samples):
                if W[i,j] > 0:
                    G.add_edge(i, j, weight=W[i,j])
        
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
        
        # Connect source to positive examples
        for idx in pos_idx:
            G.add_edge('s', idx, weight=np.inf)
            
        # Connect negative examples to sink
        for idx in neg_idx:
            G.add_edge(idx, 't', weight=np.inf)

    def predict(self):
        
        return self.labels_