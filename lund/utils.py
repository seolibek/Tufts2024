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
    def __init__(self, sigma = 1.0, DiffusionNN = 100, NEigs = 1000):
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
        
        print(Dist)
        for i in range(n):
            idx = np.argpartition(Dist[i, :], self.DiffusionNN + 1)[:self.DiffusionNN + 1]
            D_sorted = Dist[i, idx]
            sorting = np.argsort(D_sorted)
            idx = idx[sorting]

            W[i, idx[1:]] = np.exp(-(D_sorted[sorting][1:] ** 2) / (self.sigma ** 2))
            D[i, i] = np.sum(W[i, :])  
            P[i, idx[1:]] = W[i, idx[1:]] / D[i, i]

            
        pi = np.diag(D) / np.sum(np.diag(D))

        #ok do the eigendecomp here..
        print('entering try')
        try:
            
            if self.NEigs is not None:
                n_eigs = min(self.NEigs, n)
                eigvals, eigvecs = eigs(P, k = n_eigs) 
                eigvals = np.real(eigvals)
                sorted_eigvals = np.sort(-np.abs(eigvals))
                eiggap = np.abs(np.diff(sorted_eigvals)) 
                
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
                if n_eigs < 5:
                    n_eigs = 5
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
    
    #code taken from Kabir's implementation of LAND components, in python.

    # def gaussian_kernel(D):  # D = adjacency
    #     sigma = 1 * np.mean(D[D > 0])
    #     gaussian_kernel = np.exp(- (D ** 2 / (sigma ** 2)))
    #     gaussian_kernel[D == 0] = 0
    #     I = np.eye(len(D))
    #     gaussian_kernel = gaussian_kernel + I
    #     return gaussian_kernel

    # def diffusion_map(W):  # W = weight/gaussian kernel matrix
    #     row_sums = np.sum(W, axis=1)
    #     D = W / row_sums[:, np.newaxis]
    #     return D

    # def embed(P, t):  # P = diffusion matrix, t = time
    #     sparse_P = csr_matrix(P)
    #     k = 8  # Number of singular values to compute
    #     U, S, VT = svds(sparse_P, k=k)
    #     paired_elements = np.array([U[:, i] * (S[i] ** t) for i in range(k)]).T
    #     return paired_elements, U, S, VT

    # def diffusion_dist(Emb):
    #     return cdist(Emb, Emb)

    # def diffusion_distance(G, t):
    #     eigvecs = G['EigenVecs']
    #     eigvals = G['EigenVals']
    #     emb = np.array([eigvecs[:, i] * (eigvals[i] ** t) for i in range(len(eigvals))]).T
    #     return cdist(emb, emb)
    
def diffusion_distance(G,t):
# Compute the embedding
    eigenvecs = G['EigenVecs']
    eigenvals = G['EigenVals'] 
    emb = np.array([eigenvecs[:, i] * (eigenvals[i] ** t) for i in range(len(eigenvals))]).T
    # Compute the pairwise distances
    return cdist(emb, emb), emb
