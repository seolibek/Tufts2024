import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigs
#matlabafdsdlfdklfjdslkfjdljksfljkdslfkjdslf

class GraphExtractor:
    '''
    Produces graph structure to be used in cluster analysis. Graph it constructs is KNN 
    graph with Gaussian kernel edge weights

    If spatial params provided, computes modified graph that incorporates spatial geometry.
    Otherwise standard KNN

    If NEigs not included in Hyperparam structure, , = argmax(abs(diff(eigenvals))) 
    eigenvalues are included in graph structure.
    '''
    def __init__(self, sigma, DiffusionNN, NEigs = None):
        self.sigma = sigma
        self.DiffusionNN = DiffusionNN
        self.NEigs = NEigs

    def extract_graph(self, X, Dist = None):
        n = len(X)
        if Dist == None:
            Dist = pdist(X)
            Dist = squareform(Dist)
        #not doing the sptial params stuff for now, see lines 51-105
        W = np.zeros((n,n))
        P = np.zeros((n,n))
        D = np.zeros((n,n))

        for i in range(n):
            #note mink in MATLAB returns k smallest elements of array and indices.
            #rewritten in python .. sort should sort the distances and argsort should return the indices.

            #apparently there exists a more effecient way. fix later?
            D_sorted, sorting = np.sort(Dist[i, :])[:self.DiffusionNN+1], np.argsort(Dist[i, :])[:self.DiffusionNN+1]

            W[i, sorting[1:]] = (np.exp(-(D_sorted[1:] ** 2) / (self.sigma ** 2)))
            D[i, i] = np.sum(W[i, :])
            P[i, sorting[1:]] = (W[i, sorting[1:]] / D[i, i])

        pi = (np.diag(D) / np.sum(np.diag(D)))




        #ok do the eigendecomp here..
        try:
            if self.NEigs is not None:
                #worry about implementing this later
                pass
            else:
                eigvals, eigvecs = eigs(P, k=20) #scipy order is different (see docs if needed)
                eigvals = np.real(eigvals)
                sorted_eigvals = np.sort(-np.abs(eigvals))
                eiggap = np.abs(np.diff(sorted_eigvals))
                n_eigs = np.argmax(eiggap) + 1 #python indexing is 0.
                #"fringe cases"
                if n_eigs < 5:
                    n_eigs = 5
            #again,,,,,  capabilities python j doesnt have.
            idx = np.argsort(-np.abs(eigvals))
            eigvals = eigvals[idx][:n_eigs]
            eigvecs = eigvecs[:, idx][:n_eigs]
            
            # setting theoretical val for first eigenpair
            eigvecs[:, 0] = 1
            eigvals[0] = 1

            #store in graph structure?
            graph = {
                'Hyperparameters': {
                    'Sigma': self.sigma,
                    'DiffusionNN': self.DiffusionNN
                },
                'EigenVecs': np.real(eigvecs),
                'EigenVals': np.real(eigvals),
                'StationaryDist': pi,
                'P': P,
                'W': W
            }
        except Exception as e:  
            print('EigenDecomposition of P failed')
            graph = {
                'Hyperparameters': {
                    'Sigma': self.sigma,
                    'DiffusionNN': self.DiffusionNN
                },
                'EigenVecs': np.nan,
                'EigenVals': np.nan,
                'StationaryDist': np.nan,
                'P': np.nan,
                'W': np.nan
            }
        
        return graph

class DensityEstimator:
    '''
    Computes local densities of the points in an n x D matrix X, using the threshold value + KNN. 
    Matrix of distances can be passed in..
    Rewritten in python from https://github.com/sampolk/MultiscaleDiffusionClustering
    '''
    def __init__(self, DensityNN, Sigma0):
        self.DensityNN = DensityNN
        self.Sigma0 = Sigma0

    def kde(self, X, D=None):
        """
        Computes the local densities of the points in an n x D matrix X using
        the threshold value th. Uses K nearest neighbor. A matrix of distances may
        be passed in.

        :param X: Data matrix (n x D)
        :param D: Optional precomputed distance matrix
        :return: p: Local densities of points in X
        """
        #Extract Hyperparams
        NN = self.DensityNN
        sigma0 = self.Sigma0
        n = len(X)

        # Initialize 
        p = np.zeros(n)

        if D is None:
            D = pdist(X)
            D = squareform(D) #pairwise dist between points in X
        
        D = np.sort(D, axis=0)

        if NN < n:
            p = np.sum(np.exp(-(D[:NN + 1, :] ** 2) / (sigma0 ** 2)), axis=0)
            print("Updated")
        else:
            p = (np.sum(np.exp(-(D ** 2) / (sigma0 ** 2)), axis=0))

        p = (p / np.sum(p))
        return p
