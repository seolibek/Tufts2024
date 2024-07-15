import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigs
from sklearn.neighbors import KernelDensity


class GraphExtractor:
    '''
    Produces graph structure to be used in cluster analysis. Graph it constructs is KNN 
    graph with Gaussian kernel edge weights

    If spatial params provided, computes modified graph that incorporates spatial geometry.
    Otherwise standard KNN

    If NEigs not included in Hyperparam structure, , = argmax(abs(diff(eigenvals))) 
    eigenvalues are included in graph structure.
    '''
    def __init__(self, sigma = 1.0, DiffusionNN = 250, NEigs = 1000):
        # values for SalinasA; sigma = 1.0, NN = 250
        self.sigma = sigma
        self.DiffusionNN = DiffusionNN
        self.NEigs = NEigs

    def extract_graph(self, X, Dist = None):
        n = len(X)
        # print("n:",n) - value correct
        if Dist == None:
            Dist = pdist(X)
            Dist = squareform(Dist)
 
        W = np.zeros((n,n))
        P = np.zeros((n,n))
        D = np.zeros((n,n))
        
        print(Dist)
        for i in range(n):
            #note mink in MATLAB returns k smallest elements of array and indices.
            #rewritten in python .. sort should sort the distances and argsort should return the indices.

            #apparently there exists a more effecient way. fix later?
            idx = np.argpartition(Dist[i, :], self.DiffusionNN + 1)[:self.DiffusionNN + 1]
            D_sorted = Dist[i, idx]
            sorting = np.argsort(D_sorted)
            idx = idx[sorting]

            W[i, idx[1:]] = np.exp(-(D_sorted[sorting][1:] ** 2) / (self.sigma ** 2))
            D[i, i] = np.sum(W[i, :])  
            P[i, idx[1:]] = W[i, idx[1:]] / D[i, i]

            


            #TODO: Checking all of these values...
        pi = np.diag(D) / np.sum(np.diag(D))

        #ok do the eigendecomp here..
        print('entering try')
        try:
            # If the number of eigenvalues is specified, choose that.
            if self.NEigs is not None:
                #worry about implementing this later
                # there are 10 eigs
                print("SELF.NEIGS IS NOT NONE")
                print("Eigenvalues shape:", eigvals.shape)
                print("Eigenvectors shape:", eigvecs.shape)
                n_eigs = self.NEigs
                # eigvals, eigvecs = eigs(P, k=n_eigs) 
                # eigvals = np.real(eigvals)
                # sorted_eigvals = np.sort(-np.abs(eigvals))
                # eiggap = np.abs(np.diff(sorted_eigvals)) 
                
                eigvals, eigvecs = eigs(P, k=n_eigs)
                idx = np.argsort(np.abs(eigvals))[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                # pass
            else:
                print('else condition of try executed')
                eigvals, eigvecs = eigs(P, k=15) #scipy order is different (see docs if needed)
                print("Eigenvalues shape:", eigvals.shape)
                print("Eigenvectors shape:", eigvecs.shape)

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
            print("Eigenvalues shape:", eigvals.shape)
            print("Eigenvectors shape:", eigvecs.shape)

            
            # setting theoretical val for first eigenpair
            eigvecs[:, 0] = 1
            eigvals[0] = 1
            print('sucessfully')
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

class KDE:
    '''
    SKLEARN!!!!
    '''
    def __init__(self, sigma = 1.0, DiffusionNN = 250, NEigs=10):
        # values for SalinasA; sigma = 1.0, NN = 250
        self.sigma = sigma
        self.DiffusionNN = DiffusionNN
        self.NEigs = NEigs

    def kde(self, X):
        """
        Computes the local densities of the points in an n x D matrix X using
        the threshold value th. Uses K nearest neighbor. A matrix of distances may
        be passed in.

        :param X: Data matrix (n x D)
        :param D: Optional precomputed distance matrix
        :return: p: Local densities of points in X
        """
        kde_sklearn = KernelDensity(kernel='gaussian', bandwidth=self.sigma).fit(X)
        log_density = kde_sklearn.score_samples(X)
        p_sklearn = np.exp(log_density)

        p_sklearn /= np.sum(p_sklearn)
        return p_sklearn