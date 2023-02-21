import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import copy as cp
import subprocess
import os

#import pynndescent as pynn

# for classifiers in this python file 
# we use approx knn instead of knn to do sparsification
# we use a python library called pynndescent

class ParametricHNC(BaseEstimator, ClassifierMixin):
    
    ## without confidence label
    ## all labeled samples must be in the source/sink set as given

    def __init__(self, list_lambda, k=15, neighboring=True, 
                 distance_metric='euclidean', weight='RBF', epsilon=1, add_neighbor=True, adjust_lambda=False):
        self.epsilon = epsilon
        self.list_lambda = sorted(list_lambda)
        self.k = k
        self.neighboring = neighboring
        self.distance_metric = distance_metric
        self.weight = weight
        self.add_neighbor = add_neighbor
        self.adjust_lambda = adjust_lambda
        self.adjust_weight = None # we use the sum of weights in the graph
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
        # if there are unlabeled samples that do not have any labeled neighbors, we connect them to their closest labeled neighbors
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
        distances= squareform(pdist(X, metric=self.distance_metric))
        distances /= np.sqrt(X.shape[1])
        return distances
    
    def _construct_graph(self, sim, is_neighbor):
        
        n_train, n_samples = self.X_train.shape[0], sim.shape[0]
        n_test = n_samples - n_train 
        
        list_lambda = self.list_lambda
        if self.adjust_lambda:
            list_lambda /= self.adjust_weight
        num_lambda = len(list_lambda)
        
        source_index, sink_index = n_test+1, n_test+2
        lines = ["","n "+str(int(source_index))+" s", "n "+str(int(sink_index))+" t"]

        # first, get the weights of source and sink adjacent arcs
        
        pos_booleans = self.y_train == 1
        idx_positives = np.arange(n_train)[pos_booleans]
        idx_negatives = np.arange(n_train)[np.invert(pos_booleans)]
        idx_unlabeled = np.arange(n_train, n_samples)
        
        source_weights = sim[idx_unlabeled, :][:, idx_positives].sum(axis=1)
        sink_weights = sim[idx_unlabeled, :][:, idx_negatives].sum(axis=1)
        # adjust sink/source adjacent weights to optimize runtime of the cut algorithm
        min_weights = np.minimum(source_weights, sink_weights)
        source_weights -= min_weights
        sink_weights -= min_weights
        # compute the weighted degree, which would be multiplied by lambda values later
        deg_weights = sim[idx_unlabeled, :].sum(axis=1)

        for i in range(n_test):
            s_source = "a "+str(int(source_index))+" "+str(int(i+1))
            s_sink  = "a "+str(int(i+1))+" "+str(int(sink_index))
            for l in list_lambda:
                if l > 0:
                    s_source += " "+str(source_weights[i]+(l*deg_weights[i]))
                    s_sink += " "+str(sink_weights[i])
                else:
                    s_source += " "+str(source_weights[i])
                    s_sink += " "+str(sink_weights[i]+((-1)*l*deg_weights[i]))
            lines.append(s_source)
            lines.append(s_sink)
        
        if self.neighboring:
            for i in range(n_test-1):
                for j in range(i+1,n_test-1):
                    if is_neighbor[(i+n_train),(j+n_train)] or is_neighbor[(i+n_train),(j+n_train)]:
                        mutual_weight = 0.5*(sim[i+n_train,j+n_train]+sim[j+n_train,i+n_train])
                        s = "a "+str(int(i+1))+" "+str(int(j+1))+" "+str(mutual_weight)
                        lines.append(s)
                        s = "a "+str(int(j+1))+" "+str(int(i+1))+" "+str(mutual_weight)
                        lines.append(s)
        else:
            for i in range(n_test-1):
                for j in range(i+1,n_test-1):
                    s = "a "+str(int(i+1))+" "+str(int(j+1))+" "+str(0.5*(sim[i+n_train,j+n_train]+sim[j+n_train,i+n_train]))
                    lines.append(s)
                    s = "a "+str(int(j+1))+" "+str(int(i+1))+" "+str(0.5*(sim[i+n_train,j+n_train]+sim[j+n_train,i+n_train]))
                    lines.append(s)
        
        numarcs = len(lines) - 3
        lines[0] = "p par-max "+str(int(n_test+2))+" "+str(int(numarcs))+" "+str(int(num_lambda))
        
        with open("/parametric_cut/parametric_cut_input.txt", "w") as file:
            file.writelines("%s\n" % l for l in lines)
            file.close()
        
        return lines
    
    def _solve_cut(self, graph, n_test):
        #cut, partition = nx.minimum_cut(G, n_unlabeled, n_unlabeled + 1)
        breakpoints, cuts, info = pseudoflow.hpf(graph, -1, -2, const_cap="const")
        labels = np.zeros(n_test, dtype=int)
        for i in range(n_test):
            labels[i] = int(cuts[i][0])
        return labels
    
    def _solve_parametric_cut(self, n_test):
        subprocess.call(["gcc", "/parametric_cut/pseudopar-zh.c"])
        tmp=subprocess.call("/parametric_cut/a.out")
        
        file1 = open('/parametric_cut/parametric_cut_output.txt', 'r')
        lines = file1.readlines()
        file1.close()
        
        pred_arr = np.zeros((n_test,len(self.list_lambda)))
        
        for line in lines[:-2]:
            L = line.split()
            pred_arr[int(L[1])-1,int(L[2])-1:] = 1
            
        os.remove('/parametric_cut/parametric_cut_output.txt')
        
        return pred_arr

    def predict(self, X_test):
        
        X_all = np.concatenate((self.X_train, X_test), axis=0)
        
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

        ##### create text input to parametric cut rather than a networkx graph
        
        lines = self._construct_graph(similarities, is_neighbor)

        # and use the parametric cut rather than hpf
        pred_arr = self._solve_parametric_cut(X_test.shape[0])
               
        return pred_arr
