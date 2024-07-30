from sklearn.neighbors import KernelDensity
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
# Curr goal: write LUND algorithm based on proposed algorithm in LUND paper.
# With a working LUND scheme, incorporate autoencoder model into here:
# (assuming) my autoencoder can learn some accurate representation of my data,
# use the latent representation of the lower dim data to build some markov 
# transition matrix P.
# Then, go through entire scehma, which outputs the cluster assignments for the 
# lower dimensional data. 
# Afterwards, take cluster labeling, compute some centroid of points assigned to that cluster
# in the latent space. Then, use decoder to decode the centroids back to high dim space.

#Working with algorithm 2 for now, with 7 labels from 0 - 6.

def LUND(X, sigma_0, sigma, t, K):
    '''
    The implementation of LUND follows the proposed algorithm 2 in the Learning by Unsupervised Nonlinear Diffusion 
    paper.

    :param X: Data matrix
    :param sigma_0: Kernel Density Bandwidth
    :param sigma: Diffusion scaling parameter
    :param t: (time parameter)
    :param K: number of clusters
    '''

    #currently workign with salinas A, so we know K has to be 7 JUST FOR NOW. FOR NOW
    #step 1: construct markov transition matrix P using scale param sigma, referring to 'Diffusion maps' by Coifman + Lafon 
    
    #gaussian kernel
    Dist = squareform(pdist(X, 'euclidean'))
    K = np.exp(-(Dist) ** 2 / ( 2 * sigma ** 2))
    d = np.sum(K, axis = 1)
    P = K/d[:, np.newaxis] #np broadcasting stuff, divides each element in K(x,y) by corresponding d(x)

    #line 2: computing KDE p(x) for all x in X using the kernel bandwidth sigma_0
    kde = KernelDensity(bandwidth=sigma_0)
    kde.fit(X)
    log_density = kde.score_samples(X)
    p_x = np.exp(log_density)
    #line 3: computing rho_t(x) - do i need to do this? confirm, prob not
    #line 4: compute D_t(x), according to formula 2.2 in the paper
    eigenvalues, eigenvectors = eigh(P)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    n = len(X)
    D_t = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D_t[i][j] = np.sqrt(np.sum([eigenvalues[l] **(2 * t) * (eigenvectors[i,l] - eigenvectors[j,l]) ** 2 for l in range(n)]))
    #line 5: sort X according to D_t
    sorted_indices = np.argsort(D_t)[::-1]
    X_sorted = X[sorted_indices]
    #line 6 assign Y
    Y = np.zeros(len(X), dtype=int) #labels. has to be int!
    Y[sorted_indices[:K]] = range(1, K+1)
    #line 7
    sorted_indices_p = np.argsort(p_x)[::-1]
    X_sorted_p = X[sorted_indices_p]
    #lines 8-12
    for i in range(len(X)):
        if Y[sorted_indices_p[i]] == 0:
            x_star = np.argmin([D_t[sorted_indices_p[i], j] for j in range(len(X)) if p_x[j] >= p_x[sorted_indices_p[i]] and Y[j] != 0])
            Y[sorted_indices_p[i]] = Y[sorted_indices_p[x_star]]
    
    return Y

    