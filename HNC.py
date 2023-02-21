import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import pdist, squareform
import pseudoflow

class HNC(BaseEstimator, ClassifierMixin):

    def __init__(self, lambda_parameter=0, 
                 discount_factor=1, k_frac=0.1, neighboring=True, 
                 distance_metric='euclidean', weight='RBF', epsilon=1):
        self.epsilon = epsilon
        self.lambda_parameter = lambda_parameter
        self.discount_factor = discount_factor
        self.k_frac = k_frac
        self.neighboring = neighboring
        self.distance_metric = distance_metric
        self.weight = weight

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def _get_neighbor(self, dist_arr, numneigh=10):
        
        # input: dist_arr is a distance matrix of size (N, M)
        # output: a boolean matrix B of size (N,M) 
        # where B_ij = True indicates that j is among the k-th nearest neighbor of i
        # but the k-nearness is not necessarily mutual (B is not necessarily a symmetric matrix)
        
        # this method won't be used if we choose not to sparsify the graph (self.neighboring=False)
        
        boo_arr = np.full(dist_arr.shape, False)
        idx = np.argpartition(dist_arr, (1,(numneigh+1)))[:,1:(numneigh+1)]
        boo_arr[np.full((numneigh,dist_arr.shape[0]), np.arange(dist_arr.shape[0])).T, idx] = True
        
        return boo_arr
    
    
    def _get_weight(self, dist):
        # input: distance matrix
        # output: edge weight on the graph
        w_metric = self.weight
        epsilon = self.epsilon
        if w_metric == 'RBF':
            w = np.exp(-(dist**2 / (2 * epsilon ** 2)))
        elif w_metric == 'inverse':
            w = 1/(1 + dist**epsilon)
        return w

    def predict(self, X_test):

        epsilon = self.epsilon
        lambda_parameter = self.lambda_parameter
        discount_factor = self.discount_factor
        X_train = self.X_train
        y_train = self.y_train
        neighboring = self.neighboring
        
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        n_samples = n_train+n_test
        
        k_frac = self.k_frac
        if k_frac < 1:
            k = int(k_frac*(n_train+n_test))
        else:
            k = k_frac

        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, np.full(n_test, -1)))

        # Compute similarities
        distances = squareform(pdist(X, metric=self.distance_metric))
        
        #distances /= np.sqrt(X_train.shape[1])
        
        if neighboring:
            is_neighbor = self._get_neighbor(distances, numneigh=k)
            similarities = np.zeros(distances.shape)
            similarities[is_neighbor] = self._get_weight(distances[is_neighbor])
        else:
            similarities = self._get_weight(distances)

        # Get source-adjacent weights
        pos_booleans = y_train == 1
        idx_positives = np.arange(n_train)[pos_booleans]
        idx_negatives = np.arange(n_train)[np.invert(pos_booleans)]
        idx_unlabeled = np.arange(n_train, n_samples)
        
        source_weights = similarities[idx_unlabeled, :][:, idx_positives].sum(axis=1)
        source_weights += similarities[idx_unlabeled, :][:, 0:n_train].sum(axis=1) * discount_factor * lambda_parameter
        source_weights += similarities[idx_unlabeled, :][:, n_train:].sum(axis=1) * lambda_parameter

        # Get sink-adjacent weights
        sink_weights = similarities[idx_unlabeled, :][:, idx_negatives].sum(axis=1)

        # Adjust source and sink weights
        min_weights = np.minimum(source_weights, sink_weights)
        source_weights -= min_weights
        sink_weights -= min_weights

        # Create directed graph
        G = nx.DiGraph()

        if neighboring:
            for i in range(n_test-1):
                for j in range(i+1, n_test):
                    if is_neighbor[(i+n_train),(j+n_train)] or is_neighbor[(i+n_train),(j+n_train)]:
                        G.add_edge(i, j, const=similarities[(i+n_train),(j+n_train)]+similarities[(j+n_train),(i+n_train)])
                        G.add_edge(j, i, const=similarities[(i+n_train),(j+n_train)]+similarities[(j+n_train),(i+n_train)])
        else:
            for i in range(n_test-1):
                for j in range(i+1, n_test):
                    G.add_edge(i, j, const=similarities[(i+n_train),(j+n_train)]+similarities[(j+n_train),(i+n_train)])
                    G.add_edge(j, i, const=similarities[(i+n_train),(j+n_train)]+similarities[(j+n_train),(i+n_train)])
            
        # Add source/sink adjacent edges
        # the index of the source is -1
        # the index of the sink is -2
        for i in range(n_test):
            G.add_edge(-1, i, const=source_weights[i])
            G.add_edge(i, -2, const=sink_weights[i])

        # Perform cut using HPF
        #cut, partition = nx.minimum_cut(G, n_unlabeled, n_unlabeled + 1)
        breakpoints, cuts, info = pseudoflow.hpf(
        G,  # Networkx directed graph.
        -1,  # Node id of the source node.
        -2,  # Node id of the sink node.
        const_cap="const"  # Edge attribute with the constant capacity.
        )
        labels = np.zeros(n_test)
        for i in range(n_test):
            labels[i] = cuts[i][0]
        
        return labels

