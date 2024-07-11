import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.spatial.distance import pdist, squareform
from lund.lund import LearningbyUnsupervisedNonlinearDiffusion
from lund.utils import GraphExtractor,DensityEstimator
from model.utils import loadHSI,calculate_aligned_accuracy

data_path, gt_path, data_name, gt_name = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/Salinas_corrected.mat', '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/Salinas_gt.mat', 'salinas_corrected', 'salinas_gt'


X, M, N, D, HSI, GT, Y, n, K = loadHSI(data_path, gt_path, data_name, gt_name)

GT = GT - 1
print(X.shape)
HSI = X.reshape((M, N, D))
graph_extractor = GraphExtractor(sigma=1.0, DiffusionNN=10, NEigs=5) #should probably add like sucess print statements here bc idrk if this works just yet..
density_estimator = DensityEstimator(DensityNN=10, Sigma0=1.0)
#print statements are saying that the init is okay. problem might be in kde and extract graph.

p = density_estimator.kde(X)
print('kde is ok. problem is extracting graph.')
G = graph_extractor.extract_graph(X)

t = 1.0

C, K, Dt = LearningbyUnsupervisedNonlinearDiffusion(X, t, G, p)

print("Cluster labels:", C)
print("Number of clusters:", K)
print("Diffusion distances:", Dt)

accuracy = calculate_aligned_accuracy(GT, C)
print("Aligned Accuracy:", accuracy)

#note potentially worth running in an env. i cant
