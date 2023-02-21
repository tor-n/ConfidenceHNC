import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import pdist, cdist, squareform
import pseudoflow
from sklearn.neighbors import NearestNeighbors
import copy as cp

class CHNC(BaseEstimator, ClassifierMixin):
    
    ## similar to LCSNC except that we are allowed to have both labeled and unlabeled in the training set
    ## mimic the semi-supervised manner of other SSL methods

    def __init__(self, lambda_parameter=2**(-7), confidence_coef = -1,
                 k=5, k_weight=None, neighboring=True, 
                 distance_metric='euclidean', weight='RBF', epsilon=1, 
                 confidence_function='knn', confidence_weights=None, scale_lambda=True, adjust_lambda=False):
        self.epsilon = epsilon
        self.lambda_parameter = lambda_parameter
        self.confidence_coef = confidence_coef
        self.k = k
        if k_weight == None:
            self.k_weight = k
        else: 
            self.k_weight = k_weight
        self.neighboring = neighboring
        self.distance_metric = distance_metric
        self.weight = weight
        self.confidence_function = confidence_function
        self.scale_lambda = scale_lambda
        self.confidence_weights = confidence_weights
        self.distance_train = None # pairwise distance matrix of training samples
        self.add_neighbor = True
        self.adjust_lambda = adjust_lambda
        self.adjust_weight = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train        
        # compute pairwise distances of training samples
        distances = squareform(pdist(X_train, metric=self.distance_metric))
        self.distance_train = distances
        if y_train[-1] >= 0:
            self.numlabeled = len(y_train)
        else:
            self.numlabeled = next(idx for idx,i in enumerate(y_train) if i < 0)
        return self
    
    def fit_confidence(self, c):
        self.confidence_weights = c
        return self
    
    def _get_confidence(self, neigh_arr, similarities=None):
        
        confidence_function = self.confidence_function
        y_train = self.y_train
        n_train = self.X_train.shape[0]
        numlabeled = self.numlabeled
        confidence = np.zeros(numlabeled)
        k = self.k
        
        if confidence_function == 'knn':
            for i in range(numlabeled):
                #neigh_pos = np.mean(y_train[:numlabeled][neigh_arr[i,:numlabeled]])
                #confidence[i] = neigh_pos/k - 0.5
                confidence[i] = np.mean(y_train[:numlabeled][neigh_arr[i,:numlabeled]])
                if y_train[i] == 0:
                    confidence[i] = 1-confidence[i]
                confidence[i] += 0.1
                confidence[i] = 2*(confidence[i]**2)
        elif confidence_function == 'w-knn':
            for i in range(numlabeled):
                neigh_sim = similarities[i,:numlabeled][neigh_arr[i,:numlabeled]]
                pos_neigh_sim = np.dot(y_train[neigh_arr[i,:numlabeled]], neigh_sim)
                confidence[i] = sum(pos_neigh_sim)/sum(neigh_sim) - 0.5
                if y_train[i] == 0:
                    confidence[i] = (-1)*confidence[i]
                confidence[i] = 0.5 + confidence[i] 
        elif confidence_function == "constant":
            confidence = 0.5*np.ones(numlabeled)
        
        return confidence
    
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
    
    def _get_distance(self, X_test):
        #X_train = self.X_train
        n_samples = self.X_train.shape[0] + X_test.shape[0]
        n_train = self.X_train.shape[0]
        distances = np.zeros((n_samples, n_samples))
        distances[:,n_train:n_samples][n_train:n_samples,:] = squareform(pdist(X_test, metric=self.distance_metric))
        distances[n_train:n_samples,:][:,:n_train] = cdist(X_test, self.X_train, metric=self.distance_metric)
        distances[:n_train,:][:,n_train:n_samples] = distances[n_train:n_samples,:][:,:n_train].T
        distances[:n_train,:][:,:n_train] = self.distance_train
        distances /= np.sqrt(self.X_train.shape[1])
        return distances
        
    
    def _get_weight(self, dist):
        w_metric = self.weight
        epsilon = self.epsilon
        if w_metric == 'RBF':
            w = np.exp(-(dist**2 / (2 * epsilon ** 2)))
        elif w_metric == 'inverse':
            w = 1/(1 + dist**epsilon)
        return w
    
    def _construct_graph(self, sim, is_neighbor, conf_w):
        
        n_samples = sim.shape[0]
        n_labeled = self.numlabeled
        
        g = nx.DiGraph()
        for i in range(n_labeled):
            if self.y_train[i] == 1:
                g.add_edge(-1, i, const=conf_w[i])
            else:
                g.add_edge(i, -2, const=conf_w[i]) 
                
        if self.adjust_lambda:
            lambda_parameter = self.lambda_parameter/self.adjust_weight
        else:
            lambda_parameter = self.lambda_parameter
                
        if self.lambda_parameter > 0:
            source_weights = sim[n_labeled:n_samples, :].sum(axis=1) * lambda_parameter
            for i in range(n_samples-n_labeled):
                # connect unlabeled samples to source/sink
                g.add_edge(-1, i+n_labeled, const=source_weights[i])
        else:
            sink_weights = sim[n_labeled:n_samples, :].sum(axis=1) * lambda_parameter * (-1)
            for i in range(n_samples-n_labeled):
                # connect unlabeled samples to source/sink
                g.add_edge(i+n_labeled, -2, const=sink_weights[i])
                
        if self.neighboring:
            for i in range(n_samples-1):
                for j in range(i+1, n_samples):
                    # connect each pair if they are neigbors (if the relationship is mutual, the weight is stronger)
                    # we do this for both labeled and unlabeled samples
                    if is_neighbor[i,j] or is_neighbor[j,i]:
                        g.add_edge(i, j, const=0.5*(sim[i, j]+sim[j, i]))
                        g.add_edge(j, i, const=0.5*(sim[i, j]+sim[j, i]))
        else:
            for i in range(n_samples-1):
                for j in range(i+1, n_samples):
                    g.add_edge(i, j, const=sim[i, j])
                    g.add_edge(j, i, const=sim[j, i])
        return g
        
    def _solve_cut(self, graph, n_test):
        #cut, partition = nx.minimum_cut(G, n_unlabeled, n_unlabeled + 1)
        breakpoints, cuts, info = pseudoflow.hpf(graph, -1, -2, const_cap="const")
        labels = np.zeros(len(self.y_train)+n_test, dtype=int)
        for i in range(len(self.y_train)+n_test):
            labels[i] = int(cuts[i][0])
        return labels[0:len(self.y_train)], labels[len(self.y_train):]

    def predict(self, X_test):

        k = self.k
        X_train = self.X_train
        
        n_train, n_test = X_train.shape[0], X_test.shape[0]
        n_samples = n_train+n_test
        X_all = np.concatenate((X_train, X_test), axis=0)

        if k < 1:
            k = int(k*n_samples)

        if self.neighboring:
            is_neighbor, distances = self._get_neighbor(X_all, numneigh=k)
            similarities = np.full(distances.shape, 0.0)
            similarities[is_neighbor] = self._get_weight(distances[is_neighbor])
        else:
            distances = self._get_distance(X_all)
            similarities = self._get_weight(distances)

        if self.confidence_weights is None:
            print("confidence weight not given; computing confidence weight")
            # this is for the case where the confidence weights are not provided, which is the default setting
            self.confidence_weights = self._get_confidence(is_neighbor, similarities)
 
        if self.scale_lambda:
            k_lambda = k
        else:
            k_lambda = 1

        if self.confidence_coef > 0:
            coef = self.confidence_coef
        else:
            coef = similarities[np.nonzero(similarities)].mean()*(-1)*self.confidence_coef

        if self.adjust_lambda:
            self.adjust_weight = similarities[np.nonzero(similarities)].sum()
            
        c = self.confidence_weights*coef*k_lambda
        
        # construct graph
        G = self._construct_graph(similarities, is_neighbor, c)

        self.predicted_graph = G

        # Perform cut using HPF
        labels_train, labels_test = self._solve_cut(G, n_test)
        
        self.train_labels = labels_train
        
        return labels_test

    def _get_train_labels(self):
        return self.train_labels
    
