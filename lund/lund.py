import numpy as np
from scipy.spatial.distance import pdist, squareform

'''The key part of LUND is:
1.). Build a graph.
2.). Compute diffusion distances on the graph and a kernel density estimator.
3.). Label using the LUND scheme.'''

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
    DiffusionMap = np.zeros_like(G['EigenVecs'])
    #iterating over columns?? i think matlab is indexed from 1
    print('replicate error')
    for l in range(DiffusionMap.shape[1]):
        DiffusionMap[:, l] = (G['EigenVecs'][:, l] * (np.power(G['EigenVals'][l],t)))


    # Calculate pairwise diffusion distance at time t between points in X
    DiffusionDistance = squareform(pdist(np.real(DiffusionMap)))

    # compute rho_t(x), stored as rt
    rt = np.zeros(n)
    for i in range(n):
        if p[i] != np.max(p):
            rt[i] = np.min(DiffusionDistance[p > p[i], i])
        else:
            rt[i] = np.max(DiffusionDistance[i, :])

    # Extract Dt(x) and sort in descending order
    #ignore . element wise operations handled in python.
    Dt = rt * p
    m_sorting = np.argsort(-Dt) #sorting in descending order technically bc negative versions

    # Determine K based on the ratio of sorted Dt(x_{m_k})
    if K_known is not None: #nargin dne in python.
        K = K_known
    else:
        ratios = np.divide(Dt[m_sorting[0:n-1]], Dt[m_sorting[1:n]])
        K = np.argmax(ratios)

    if K == 1:
        C = np.ones(n, dtype=int)
    else:
        # Label modes
        C[m_sorting[:K]] = np.arange(0, K)

        # Label non-modal points according to the label of their Dt-nearest
        # neighbor of higher density that is already labeled.
        l_sorting = np.argsort(-p)
        for j in range(n):
            i = l_sorting[j]
            if C[i] == 0:  # unlabeled point
                candidates = np.where((p >= p[i]) & (C > 0))[0]
                temp_idx = np.argmin(DiffusionDistance[i, candidates])
                C[i] = C[candidates[temp_idx]]

    return C, K, Dt
