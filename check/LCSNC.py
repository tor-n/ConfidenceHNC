import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import pdist, cdist, squareform
import pseudoflow

class LCSNC(BaseEstimator, ClassifierMixin):

    def __init__(self, lambda_parameter=0, deviation_parameter=1,
                 k_frac=0.1, neighboring=True, 
                 distance_metric='euclidean', weight='RBF', epsilon=1, 
                 confidence_function='knn'):
        self.epsilon = epsilon
        self.lambda_parameter = lambda_parameter
        self.deviation_parameter = deviation_parameter
        self.k_frac = k_frac
        self.neighboring = neighboring
        self.distance_metric = distance_metric
        self.weight = weight
        self.graph = None
        self.confidence_function = confidence_function
        self.confidence_weights = None
        self.distance_train = None # pairwise distance matrix of training samples

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        G = nx.DiGraph()
        
        # compute pairwise distances of training samples
        distances = squareform(pdist(X_train, metric=self.distance_metric))
        self.distance_train = distances
        
        # connect each labeled sample to the source/sink node with some confidence weight
        c = self._get_confidence(X_train, y_train, distances)
        self.confidence_weights = c
        
        for i in range(len(y_train)):
            if y_train[i] == 1:
                G.add_edge(-1, i, const=c[i])
            else:
                G.add_edge(i, -2, const=c[i]) 
        self.graph = G
        
        return self
    
    def _get_confidence(self, X_train, y_train, distances):
        deviation_parameter = self.deviation_parameter
        confidence_function = self.confidence_function

        confidence = np.zeros(y_train.shape)
        k_frac = self.k_frac
        
        if k_frac < 1:
            k = int(k_frac*(X_train.shape[0]))
        else:
            k = k_frac
        
        ## use locality mean: find m nearest neighbors from each label and compute the mean features and compare the distances
        if confidence_function == 'localmean':

            pos_booleans = y_train == 1
            idx_positives = np.arange(X_train.shape[0])[pos_booleans]
            idx_negatives = np.arange(X_train.shape[0])[np.invert(pos_booleans)]
            
            if k >= np.min([len(idx_positives),len(idx_negatives)]):
                k = np.min([len(idx_positives),len(idx_negatives)]) - 1
            
            positive_neighbors = idx_positives[np.argpartition(distances[:,idx_positives], (1, (k+1)))[:,:(k+1)]]
            negative_neighbors = idx_negatives[np.argpartition(distances[:,idx_negatives], (1, (k+1)))[:,:(k+1)]]
            
            for i in range(X_train.shape[0]):
                if y_train[i] == 1:
                    positive_mean = np.mean(X_train[positive_neighbors[1:],:], axis = 0)
                    negative_mean = np.mean(X_train[negative_neighbors[:k],:], axis = 0)
                    positive_distance = np.linalg.norm(positive_mean - X_train[i,:])
                    negative_distance = np.linalg.norm(negative_mean - X_train[i,:])
                    positive_weight = self._get_weight(positive_distance)
                    negative_weight = self._get_weight(negative_distance)
                    if positive_weight+negative_weight > 0:
                        confidence[i] = positive_weight/(positive_weight+negative_weight)
                else:
                    positive_mean = np.mean(X_train[positive_neighbors[:k],:], axis = 0)
                    negative_mean = np.mean(X_train[negative_neighbors[1:],:], axis = 0)
                    positive_distance = np.linalg.norm(positive_mean - X_train[i,:])
                    negative_distance = np.linalg.norm(negative_mean - X_train[i,:])
                    positive_weight = self._get_weight(positive_distance)
                    negative_weight = self._get_weight(negative_distance)
                    if positive_weight+negative_weight > 0:
                        confidence[i] = negative_weight/(positive_weight+negative_weight)
                        
        elif confidence_function == 'knn':
            idx_neighbors = np.argpartition(distances, (1,(k+1)))[:,1:(k+1)]
            confidence = np.sum(y_train[idx_neighbors], axis=1)/k
            for i in range(X_train.shape[0]):
                if y_train[i] == 0:
                    confidence[i] = 1 - confidence[i]
                    
        elif confidence_function == "constant":
            confidence = np.ones(y_train.shape)
        
        confidence = k*deviation_parameter*confidence
        
        return confidence
    
    def _get_neighbor(self, dist_arr, numneigh=10):
        # use later when sparsifying graph
        boo_arr = np.full(dist_arr.shape, False)
        idx = np.argpartition(dist_arr, (1,(numneigh+1)))[:,1:(numneigh+1)]
        boo_arr[np.full((numneigh,dist_arr.shape[0]), np.arange(dist_arr.shape[0])).T, idx] = True
        return boo_arr
    
    def _get_weight(self, dist):
        w_metric = self.weight
        epsilon = self.epsilon
        if w_metric == 'RBF':
            w = np.exp(-(dist**2 / (2 * epsilon ** 2)))
        elif w_metric == 'inverse':
            w = 1/(1 + dist**epsilon)
        return w
    
    @staticmethod
    def _cut_to_label(cuts):
        # input: cuts dictionary where key = node index and values = labels for different lambda breakpoints
        # output: array of labels, size of array is (numbreakpoints, numsamples)
        n = len(cuts.keys()) - 2
        if n-1 not in cuts.keys():
            print("check")
            n = np.max(cuts.keys()) + 1
        y_array = np.array([cuts[i] for i in range(n)]).T
        return y_array
    
    @staticmethod
    def _evaluate(target, pred_arr, metric='accuracy'):
        n = len(target)
        n_bps, n_all = pred_arr.shape
        if n_all > n:
            pred_arr = pred_arr[:,-n:]
        
        if metric == 'accuracy':
            scores = [accuracy_score(target, pred_arr[i,:]) for i in range(n_bps)]
        elif metric == 'F1':
            scores = [f1_score(target, pred_arr[i,:]) for i in range(n_bps)]
        
        return scores

    def predict(self, X_test):

        epsilon = self.epsilon
        lambda_parameter = self.lambda_parameter
        X_train = self.X_train
        y_train = self.y_train
        neighboring = self.neighboring
        G = self.graph
        
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
        distances_within = squareform(pdist(X_test, metric=self.distance_metric))
        distances_test_train = cdist(X_test, X_train, metric=self.distance_metric)
        
        distances = np.zeros((n_samples, n_samples))
        distances[:,n_train:n_samples][n_train:n_samples,:] = distances_within
        distances[n_train:n_samples,:][:,:n_train] = distances_test_train 
        distances[:n_train,:][:,n_train:n_samples] = distances_test_train.T
        distances[:n_train,:][:,:n_train] = self.distance_train
        
        #distances /= np.sqrt(X_train.shape[1]) # why did we do this???
        
        if neighboring:
            is_neighbor = self._get_neighbor(distances, numneigh=k)
            similarities = np.full(distances.shape, 0.0)
            similarities[is_neighbor] = self._get_weight(distances[is_neighbor])
        else:
            similarities = self._get_weight(distances)

        # Get source-adjacent weights
        idx_unlabeled = np.arange(n_train, n_samples)
        source_weights = similarities[idx_unlabeled, :].sum(axis=1) * lambda_parameter

        # create graph
        if neighboring:
            for i in range(n_samples-1):
                for j in range(i+1, n_samples):
                    # connect each pair if they are neigbors (if the relationship is mutual, the weight is stronger)
                    # we do this for both labeled and unlabeled samples
                    if is_neighbor[i,j] or is_neighbor[j,i]:
                        G.add_edge(i, j, const=similarities[i, j]+similarities[j, i])
                        G.add_edge(j, i, const=similarities[i, j]+similarities[j, i])
        else:
            for i in range(n_samples-1):
                for j in range(i+1, n_samples):
                    G.add_edge(i, j, const=similarities[i, j])
                    G.add_edge(j, i, const=similarities[j, i])
            
        # Add source/sink adjacent edges
        # the index of the source is -1
        # the index of the sink is -2
        for i in range(n_test):
            # connect unlabeled samples to source/sink
            G.add_edge(-1, i+n_train, const=source_weights[i])
            
        #self.predicted_graph = G

        # Perform cut using HPF
        #cut, partition = nx.minimum_cut(G, n_unlabeled, n_unlabeled + 1)
        breakpoints, cuts, info = pseudoflow.hpf(
        G,  # Networkx directed graph.
        -1,  # Node id of the source node.
        -2,  # Node id of the sink node.
        const_cap="const"
        )
        
        labels = np.zeros(n_test)
        for i in range(n_test):
            labels[i] = cuts[i+n_train][0]
        
        # collect information about the results on the labeled samples
        labels_train = np.zeros(n_train)
        for i in range(n_train):
            labels_train[i] = cuts[i][0]
        
        self.train_labels = labels_train
        
        return labels
    
    def _get_train_labels(self):
        return self.train_labels

