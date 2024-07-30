import numpy as np
from scipy.spatial.distance import pdist, squareform

'''The key part of LUND is:
1.). Build a graph.
2.). Compute diffusion distances on the graph and a kernel density estimator.
3.). Label using the LUND scheme.'''



#below is Sam Polk's implementation of LUND.

def LearningbyUnsupervisedNonlinearDiffusion(X, t, G, p, K_known=None):
    '''
    This functino produces a structure with multiscale clusterings produced with the LUND algorithm.
    This code is a  python adaptation of the original code as published in 
    https://github.com/sampolk/MultiscaleDiffusionClustering/

    :param X: Data matrix
    :param t: Diffusion Time step
    :param G: Graph structure computed using extract_graph.py (also adapted from same repository)
    :param p: Kernel Density Estimator
    '''
    
    n = len(X)
    C = np.zeros(n, dtype=int)

    # Calculate diffusion map
    print('entered lund')
    print("Number of Eigenvalues:", len(G['EigenVals']))
    print("Number of Eigenvectors:", G['EigenVecs'].shape)

    DiffusionMap = np.zeros_like(G['EigenVecs'])

    print("Shape of G['EigenVecs']:", G['EigenVecs'].shape)
    print("Shape of G['EigenVals']:", G['EigenVals'].shape)
    for l in range(DiffusionMap.shape[1]):
        DiffusionMap[:, l] = G['EigenVecs'][:, l] * (G['EigenVals'][l]**t)

    DiffusionDistance = squareform(pdist(np.real(DiffusionMap)))


    # compute rho_t(x), stored as rt
    print("Shape of DiffusionDistance:", DiffusionDistance.shape)
    print("Shape of p:", p.shape)

    rt = np.zeros(n)
    for i in range(n):
        if p[i] != np.max(p):
            rt[i] = np.min(DiffusionDistance[p > p[i], i])
        else:
            rt[i] = np.max(DiffusionDistance[i, :])

    Dt = rt * p
    m_sorting = np.argsort(-Dt) 
    print("m_sorting:", m_sorting)

    if K_known is not None: 
        K = K_known
    else:
        ratios = np.divide(Dt[m_sorting[0:n-1]], Dt[m_sorting[1:n]])
        K = np.argmax(ratios) + 1
    print("Number of clusters K:", K)

    if K == 1:
        C = np.ones(n, dtype=int)
    else:
        # Label modes
        C[m_sorting[:K]] = np.arange(1, K + 1)
        l_sorting = np.argsort(-p)
        for j in range(n):
            i = l_sorting[j]
            if C[i] == 0:  # unlabeled point
                candidates = np.where((p >= p[i]) & (C > 0))[0]
                if len(candidates) > 0:
                    temp_idx = np.argmin(DiffusionDistance[i, candidates])
                    C[i] = C[candidates[temp_idx]]
                # else:
                #     # Handling the case where no candidates are found?
                #     C[i] = -1  
                #     print('this was a bandaid')

    return C, K, Dt
