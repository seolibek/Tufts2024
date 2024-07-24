import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigs
from sklearn.neighbors import KernelDensity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cdist
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize

class GraphExtractor:
    '''
    Produces graph structure to be used in cluster analysis. Graph it constructs is KNN 
    graph with Gaussian kernel edge weights

    If spatial params provided, computes modified graph that incorporates spatial geometry.
    Otherwise standard KNN

    If NEigs not included in Hyperparam structure, , = argmax(abs(diff(eigenvals))) 
    eigenvalues are included in graph structure.
    '''
    def __init__(self, sigma = 5.0, DiffusionNN = 20, NEigs = 100):
        self.sigma = sigma
        self.DiffusionNN = DiffusionNN
        self.NEigs = int(NEigs)
        print(f"Initialized with NEigs = {self.NEigs} (type: {type(self.NEigs)})")


    def extract_graph(self, X, Dist = None):
        n = len(X)
        if Dist == None:
            Dist = pdist(X)
            Dist = squareform(Dist)
            print("Pairwise Distance Matrix:")
            print(Dist)
 
        W = np.zeros((n,n))
        P = np.zeros((n,n))
        D = np.zeros((n,n))
        
        for i in range(n):
            idx = np.argpartition(Dist[i, :], self.DiffusionNN + 1)[:self.DiffusionNN + 1]
            D_sorted = Dist[i, idx]
            sorting = np.argsort(D_sorted)
            idx = idx[sorting]

            W[i, idx[1:]] = np.exp(-(D_sorted[1:] ** 2) / (self.sigma ** 2))
            D[i, i] = np.sum(W[i, :])  
            P[i, idx[1:]] = W[i, idx[1:]] / D[i, i]

            
        pi = np.diag(D) / np.sum(np.diag(D))
        print("Transition Matrix:")
        print(P)
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
        # print(P[1])
        # print('prob',P)


        # row_sums = np.sum(P, axis=1)
        # print('row sum',row_sums)
        # # Check if each row sums to 1
        # if np.allclose(row_sums, 1):
        #     print("All rows sum to 1.")
        # else:
        #     print("Not all rows sum to 1.")

        #below is a test to see if my transition matrix, P,  is constructed as expected. Also W and D.
        # X, GT = make_moons(n_samples=100, noise=0.1, random_state=42)

        # plt.scatter(X[:, 0], X[:, 1], c=GT, cmap=plt.cm.Paired)
        # plt.title('Moons Dataset with Ground Truth Labels')
        # plt.show()

        # # Plot transitions (for a subset for clarity)
        # subset = np.random.choice(range(len(X)), size=20, replace=False)
        # for i in subset:
        #     for j in range(len(P)):
        #         if P[i, j] > 0:
        #             plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'k-', alpha=0.2)

        # plt.scatter(X[:, 0], X[:, 1], c=GT, cmap=plt.cm.Paired)
        # plt.title('Moons Dataset with Transition Paths')
        # plt.show()
        # Verify the Degree Matrix
        # degree_sums = np.sum(W, axis=1)
        # print("Degree sums:", degree_sums)

        # # Visualize the Weight Matrix
        # plt.figure(figsize=(10, 8))
        # plt.imshow(W, cmap='viridis')
        # plt.colorbar()
        # plt.title('Weight Matrix (W)')
        # plt.show()

        # # Visualize the Degree Matrix
        # plt.figure(figsize=(10, 8))
        # plt.imshow(D, cmap='viridis')
        # plt.colorbar()
        # plt.title('Degree Matrix (D)')
        # plt.show()

        # # Visualize Degree Values
        # plt.figure(figsize=(10, 8))
        # plt.bar(np.arange(n), np.diag(D))
        # plt.title('Degree Values')
        # plt.xlabel('Index')
        # plt.ylabel('Degree')
        # plt.show()
        
        #ok do the eigendecomp here..
        try:
            
            if self.NEigs is not None:
                n_eigs = min(self.NEigs, n)
                eigvals, eigvecs = eigs(P, k = 20) 
                eigvals = np.real(eigvals)
                sorted_eigvals = np.sort(-np.abs(eigvals))
                eiggap = np.abs(np.diff(sorted_eigvals)) 
                
                # pass
            else:
                eigvals, eigvecs = eigs(P, k=20) #scipy order is different (see docs if needed)
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
            # Debugging output
             # Plot the eigenvalues
            # plt.figure(figsize=(10, 6))
            # plt.plot(np.arange(len(eigvals)), eigvals, 'o-', label='Eigenvalues')
            # plt.title('Eigenvalues of Transition Matrix')
            # plt.xlabel('Index')
            # plt.ylabel('Eigenvalue')
            # plt.legend()
            # plt.grid(True)
            # plt.show()

            # # Plot the eigengap
            # eiggap = np.abs(np.diff(np.sort(-np.abs(eigvals))))
            # plt.figure(figsize=(10, 6))
            # plt.plot(np.arange(len(eiggap)), eiggap, 'o-', label='Eigengap')
            # plt.title('Eigengap of Transition Matrix')
            # plt.xlabel('Index')
            # plt.ylabel('Eigengap')
            # plt.legend()
            # plt.grid(True)
            # plt.show()

            # # Plot the first few eigenvectors
            # plt.figure(figsize=(10, 6))
            # for i in range(3):  # Plot the first three eigenvectors
            #     plt.plot(eigvecs[:, i], label=f'Eigenvector {i+1}')
            # plt.title('First Three Eigenvectors')
            # plt.xlabel('Index')
            # plt.ylabel('Value')
            # plt.legend()
            # plt.grid(True)
            # plt.show()

            # # Scatter plot using the second and third eigenvectors
            # plt.figure(figsize=(10, 8))
            # plt.scatter(X[:, 0], X[:, 1], c=eigvecs[:, 1], cmap='viridis')
            # plt.colorbar(label='Eigenvector 2 values')
            # plt.title('Data Points Colored by Second Eigenvector')
            # plt.xlabel('Feature 1')
            # plt.ylabel('Feature 2')
            # plt.show()

            # plt.figure(figsize=(10, 8))
            # plt.scatter(X[:, 0], X[:, 1], c=eigvecs[:, 2], cmap='viridis')
            # plt.colorbar(label='Eigenvector 3 values')
            # plt.title('Data Points Colored by Third Eigenvector')
            # plt.xlabel('Feature 1')
            # plt.ylabel('Feature 2')
            # plt.show()



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
    
    print(f"Embedding at t={t}:")
    print(emb)

    diffusion_distances = cdist(emb, emb)

    print(f"Diffusion distance matrix at t={t}:")
    print(diffusion_distances)

    return diffusion_distances, emb
