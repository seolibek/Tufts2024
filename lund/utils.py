import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigs
from sklearn.neighbors import KernelDensity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cdist

class GraphExtractor:
    '''
    Produces graph structure to be used in cluster analysis. Graph it constructs is KNN 
    graph with Gaussian kernel edge weights

    If spatial params provided, computes modified graph that incorporates spatial geometry.
    Otherwise standard KNN

    If NEigs not included in Hyperparam structure, , = argmax(abs(diff(eigenvals))) 
    eigenvalues are included in graph structure.
    '''
    def __init__(self, sigma = 1.0, DiffusionNN = 10, NEigs = 100):
        self.sigma = sigma
        self.DiffusionNN = DiffusionNN
        self.NEigs = int(NEigs)
        print(f"Initialized with NEigs = {self.NEigs} (type: {type(self.NEigs)})")


    def extract_graph(self, X, Dist = None):
        n = len(X)
        if Dist == None:
            Dist = pdist(X)
            Dist = squareform(Dist)
 
        W = np.zeros((n,n))
        P = np.zeros((n,n))
        D = np.zeros((n,n))
        
        #potentially the indexing here
        for i in range(n):
            idx = np.argpartition(Dist[i, :], self.DiffusionNN + 1)[:self.DiffusionNN + 1]
            D_sorted = Dist[i, idx]
            sorting = np.argsort(D_sorted)
            idx = idx[sorting]

            W[i, idx[1:]] = np.exp(-(D_sorted[1:] ** 2) / (self.sigma ** 2))
            D[i, i] = np.sum(W[i, :])  
            P[i, idx[1:]] = W[i, idx[1:]] / D[i, i]

            
        pi = np.diag(D) / np.sum(np.diag(D))
        print(P.shape)
        # Dist = squareform(pdist(X))
        # W = np.zeros((n, n))
        # P = np.zeros((n, n))
        # D = np.zeros(n)  

        # for i in range(n):
        #     idx = np.argpartition(Dist[i, :], self.DiffusionNN + 1)[:self.DiffusionNN + 1]
        #     D_sorted = Dist[i, idx]
        #     sorting = np.argsort(D_sorted)
        #     idx = idx[sorting]

        #     W[i, idx[1:]] = np.exp(-(D_sorted[1:] ** 2) / (self.sigma ** 2))
            
        #     D[i] = np.sum(W[i, :])

        #     if D[i] > 0:  
        #         P[i, idx[1:]] = W[i, idx[1:]] / D[i]

        # pi = D / np.sum(D)
        print('prob',P)
        row_sums = np.sum(P, axis=1)

        # Check if each row sums to 1
        if np.allclose(row_sums, 1):
            print("All rows sum to 1.")
        else:
            print("Not all rows sum to 1.")
        #ok do the eigendecomp here..
        try:
            
            if self.NEigs is not None:
                n_eigs = min(self.NEigs, n)
                eigvals, eigvecs = eigs(P, k = n_eigs) 
                eigvals = np.real(eigvals)
                sorted_eigvals = np.sort(-np.abs(eigvals))
                eiggap = np.abs(np.diff(sorted_eigvals)) 
                
                # pass
            else:
                eigvals, eigvecs = eigs(P, k=15) #scipy order is different (see docs if needed)
                eigvals = np.real(eigvals)
                sorted_eigvals = np.sort(-np.abs(eigvals))
                eiggap = np.abs(np.diff(sorted_eigvals))
                n_eigs = np.argmax(eiggap) + 1 #python indexing is 0.
                if n_eigs < 5:
                    n_eigs = 5

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
    
    #code taken from Kabir's implementation of LAND components, in python.

def diffusion_distance(G,t):
# Compute the embedding
    eigenvecs = G['EigenVecs']
    eigenvals = G['EigenVals'] 
    emb = np.array([eigenvecs[:, i] * (eigenvals[i] ** t) for i in range(len(eigenvals))]).T
    # Compute the pairwise distances
    return cdist(emb, emb), emb

#fix and plot difufsion distance and graph extractor, then lund ? should be fine. goal today is finish lund. tomorrow work on connecting autoencdoer workflow. perhaps start writing a more complex mordel