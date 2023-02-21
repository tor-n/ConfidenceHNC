import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import pdist, cdist, squareform
import pseudoflow
from sklearn.neighbors import NearestNeighbors
import copy as cp

class SNC(BaseEstimator, ClassifierMixin):
    
    ## without confidence label
    ## all labeled samples must be in the source/sink set as given

    def __init__(self, lambda_parameter=0, k=15, neighboring=True, 
                 distance_metric='euclidean', weight='RBF', epsilon=1, add_neighbor=True, adjust_lambda=False):
        self.epsilon = epsilon
        self.lambda_parameter = lambda_parameter
        self.k = k
        self.neighboring = neighboring
        self.distance_metric = distance_metric
        self.weight = weight
        self.distance_train = None
        self.add_neighbor = add_neighbor
        self.adjust_lambda = adjust_lambda
        self.adjust_weight = None
        self.neighbors_boo = None
        self.neighbors_dist = None
        self.neighbors_fitted = False

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def fit_neighbor(self, neighbor_boo, neighbor_dist, X_test):
        self.neighbors_boo = neighbor_boo
        self.neighbors_dist = neighbor_dist
        self.neighbors_fitted = True
        
        if self.add_neighbor:
            n_train = self.X_train.shape[0]
            unpaired_samples = [i+n_train for i,j in enumerate(np.sum(self.neighbors_boo[n_train:,:n_train], axis=1) == 0) if j]
            if len(unpaired_samples) > 0:
                data_arr = np.concatenate((self.X_train, X_test), axis=0)
                neigh_added = NearestNeighbors(n_neighbors=1, metric=self.distance_metric)
                neigh_added.fit(data_arr[:n_train,:])
                neighbors_added = neigh_added.kneighbors(data_arr[unpaired_samples,:])
                for i in range(len(unpaired_samples)):
                    self.neighbors_boo[unpaired_samples[i]][neighbors_added[1][i,0]] = True
                    self.neighbors_dist[unpaired_samples[i]][neighbors_added[1][i,0]] = neighbors_added[0][i,0]/np.sqrt(self.X_train.shape[1])
        return self
    
    def _get_neighbor(self, data_arr, numneigh=10):
        
        # input: dist_arr is a distance matrix of size (N, M)
        # output: a boolean matrix B of size (N,M) 
        # where B_ij = True indicates that j is among the k-th nearest neighbor of i
        # each row of B has exactly k True's
        # but the k-nearness is not necessarily mutual (B is not necessarily a symmetric matrix)
        
        # this method won't be used if we choose not to sparsify the graph (self.neighboring=False)
        
        boo_arr = np.full((data_arr.shape[0], data_arr.shape[0]), False)
        dist_arr = np.zeros((data_arr.shape[0], data_arr.shape[0]))

        neigh = NearestNeighbors(n_neighbors=1+numneigh, metric=self.distance_metric)
        neigh.fit(data_arr)
        neighbors = neigh.kneighbors(data_arr)
        for i in range(data_arr.shape[0]):
            for ind, j in enumerate(neighbors[1][i,1:]):
                boo_arr[i][j] = True
                dist_arr[i][j] = neighbors[0][i,1+ind]
        
        # add neighbor (each unlabeled samples must be connected to at least one labeled sample)
        if self.add_neighbor:
            n_train = self.X_train.shape[0]
            unpaired_samples = [i+n_train for i,j in enumerate(np.sum(boo_arr[n_train:,:n_train], axis=1) == 0) if j]
            if len(unpaired_samples) > 0:
                neigh_added = NearestNeighbors(n_neighbors=1, metric=self.distance_metric)
                neigh_added.fit(data_arr[:n_train,:])
                neighbors_added = neigh_added.kneighbors(data_arr[unpaired_samples,:])
                for i in range(len(unpaired_samples)):
                    boo_arr[unpaired_samples[i]][neighbors_added[1][i,0]] = True
                    dist_arr[unpaired_samples[i]][neighbors_added[1][i,0]] = neighbors_added[0][i,0]
        
        dist_arr /= np.sqrt(self.X_train.shape[1])
        
        return boo_arr, dist_arr
    
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
    
    def _get_distance(self, X):
        distances = squareform(pdist(X, metric=self.distance_metric))
        distances /= np.sqrt(X.shape[1])
        return distances
    
    def _construct_graph(self, sim, is_neighbor):
        
        n_train, n_samples = self.X_train.shape[0], sim.shape[0]
        n_test = n_samples - n_train 
        
        pos_booleans = self.y_train == 1
        idx_positives = np.arange(n_train)[pos_booleans]
        idx_negatives = np.arange(n_train)[np.invert(pos_booleans)]
        idx_unlabeled = np.arange(n_train, n_samples)
        
        source_weights = sim[idx_unlabeled, :][:, idx_positives].sum(axis=1)
        sink_weights = sim[idx_unlabeled, :][:, idx_negatives].sum(axis=1)
        if self.adjust_lambda:
            lambda_parameter = self.lambda_parameter/self.adjust_weight
        else:
            lambda_parameter = self.lambda_parameter
        if self.lambda_parameter > 0:
            source_weights += sim[idx_unlabeled, :].sum(axis=1)*lambda_parameter
        else:
            sink_weights += sim[idx_unlabeled, :].sum(axis=1)*(-1)*lambda_parameter
        
        # Adjust source and sink weights
        min_weights = np.minimum(source_weights, sink_weights)
        source_weights -= min_weights
        sink_weights -= min_weights

        # Create directed graph
        g = nx.DiGraph()

        if self.neighboring:
            for i in range(n_test-1):
                for j in range(i+1, n_test):
                    if is_neighbor[(i+n_train),(j+n_train)] or is_neighbor[(i+n_train),(j+n_train)]:
                        g.add_edge(i, j, const=0.5*(sim[(i+n_train),(j+n_train)]+sim[(j+n_train),(i+n_train)]))
                        g.add_edge(j, i, const=0.5*(sim[(i+n_train),(j+n_train)]+sim[(j+n_train),(i+n_train)]))
        else:
            for i in range(n_test-1):
                for j in range(i+1, n_test):
                    g.add_edge(i, j, const=sim[(i+n_train),(j+n_train)]+sim[(j+n_train),(i+n_train)])
                    g.add_edge(j, i, const=sim[(i+n_train),(j+n_train)]+sim[(j+n_train),(i+n_train)])
            
        # Add source/sink adjacent edges
        # the index of the source is -1
        # the index of the sink is -2
        for i in range(n_test):
            g.add_edge(-1, i, const=source_weights[i])
            g.add_edge(i, -2, const=sink_weights[i])
        
        return g
    
    def _solve_cut(self, graph, n_test):
        #cut, partition = nx.minimum_cut(G, n_unlabeled, n_unlabeled + 1)
        breakpoints, cuts, info = pseudoflow.hpf(graph, -1, -2, const_cap="const")
        labels = np.zeros(n_test, dtype=int)
        for i in range(n_test):
            labels[i] = int(cuts[i][0])
        return labels

    def predict(self, X_test):

        X_train = self.X_train
        
        n_test = X_test.shape[0]
        
        k = self.k

        # Compute similarities
        X_all = np.concatenate((X_train, X_test), axis=0)
        
        if self.neighboring:
            if self.neighbors_fitted:
                is_neighbor, distances = self.neighbors_boo, self.neighbors_dist
            else:
                is_neighbor, distances = self._get_neighbor(X_all, numneigh=self.k)
            similarities = np.zeros(distances.shape)
            similarities[is_neighbor] = self._get_weight(distances[is_neighbor])
        else:
            distances = self._get_distance(X_all)
            similarities = self._get_weight(distances)
        
        if self.adjust_lambda:
            self.adjust_weight = similarities[np.nonzero(similarities)].sum()

        # graph construction
        G = self._construct_graph(similarities, is_neighbor)
        
        # Perform cut using HPF
        labels = self._solve_cut(G, n_test)
               
        return labels
