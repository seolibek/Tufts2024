#Goal: clean functions ... kmeans could be refactored, so could compare_umap (technically doesnt even just comppare umap so the name needs a change..)
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import umap
import pandas as pd
from matplotlib.animation import FuncAnimation
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

#rename whole thing as dimensionality reduction thing
def loadHSI(data_path, ground_truth_path, data_col, ground_truth_col):
  """
    Read and parse the data files, creating referrable Hyperspectral imagery data and Ground Truth data.
    Hyperspectral images are reshaped for later use.

    :param data_path: Path to the data file.
    :param ground_truth_path: Path to the ground truth file.
    :param data_col: Column name of data.
    :param ground_truth_col: Column name of ground truth.
    :return: X, M, N, D, HSI, GT, Y, n, K

    where HSI, GT are called for hyperspectral imagery and ground truth, respectively.
    """
  data = scipy.io.loadmat(data_path)
  HSI = data[data_col]

  data = scipy.io.loadmat(ground_truth_path)
  GT = data[ground_truth_col]

  M, N, D = HSI.shape
  n = M * N
  X = HSI.reshape((n, D))
  X = X.astype(np.float64)

  norms = np.sqrt(np.sum(X**2, axis = 0))
  X /= norms

  X += 1e-6 * np.random.randn(*X.shape)

  HSI = X.reshape((M, N, D))
  HSI = HSI.reshape(-1, HSI.shape[2])

  new_gt = np.zeros_like(GT)
  unique_classes = np.unique(GT)
  K = len(unique_classes)

  for k, uc in enumerate(unique_classes, start=1):
      new_gt[GT == uc] = k
  n = new_gt.size
  Y = new_gt.reshape((n, 1))
  GT = new_gt

  return X, M, N, D, HSI, GT, Y, n, K



def show_clusterable_embedding(hsi_data, ground_truth):
  """
    Projection of HSI data into a 2d mapping,for visualization purposes.
    Clusterable embeddings, generated by UMAP

    :param hsi_data: Hyperspectral Imaging Data
    :param ground_truth: Ground Truth Imaging Data
    :return: None
    """

  clusterable_embedding = umap.UMAP(
      n_neighbors=30,
      min_dist=0.0,
      n_components=2,
      random_state=42,
  ).fit_transform(hsi_data)

  plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
              c=ground_truth, s=0.1, cmap='Spectral')

  return clusterable_embedding



def visualize_umap(data_reshaped, ground_truth, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data_reshaped)
    rotation_3d(u = u, n_components=n_components,ground_truth=ground_truth,title=title)
    return u


def k_means_with_umap(dim_reduced_data, ground_truth):

  GT_flat = ground_truth.flatten()
  num_clusters = len(np.unique(GT_flat))


  # num_clusters = len(np.unique(GT_flat)) - (1 if 0 in GT_flat else 0)  # Adjust based on whether '0' should be excluded
  kmeans = KMeans(n_clusters=num_clusters, random_state=42)
  labels = kmeans.fit_predict(dim_reduced_data)

  ari = adjusted_rand_score(GT_flat, labels)
  print("UMAP Adjusted Rand Index (ARI):", ari)
  return ari, labels


def k_means_with_pca(data,n_components, ground_truth):
  GT_flat_PCA = ground_truth.flatten()

  pca = PCA(n_components)
  data_reduced = pca.fit_transform(data)

  num_clusters = len(np.unique(GT_flat_PCA))
  kmeans = KMeans(n_clusters=num_clusters, random_state=42)
  labels = kmeans.fit_predict(data_reduced)

  ari = adjusted_rand_score(GT_flat_PCA, labels)
  print("PCA Adjusted Rand Index (ARI):", ari)
  return ari, labels


def calculate_aligned_accuracy(ground_truth, cluster_labels):
    true_labels = ground_truth.flatten()
    cm = confusion_matrix(true_labels, cluster_labels)

    row_ind, col_ind = linear_sum_assignment(-cm)
    label_mapping = {cluster_label: true_label for cluster_label, true_label in zip(col_ind, row_ind)}
    aligned_labels = [label_mapping[label] for label in cluster_labels]

    accuracy = np.mean(np.array(aligned_labels) == np.array(true_labels))

    return accuracy

def compare_umap(data, ground_truth, dataset_name, compare_dim, compare_neighbors, tSNE):
    
    """
    Compares newer dimensionality reduction methods to the PCA to compare baseline performance. compare_dim and compare_neighbors 
    are flags, where 

    :param data: Hyperspectral imagery, reshaped in the preprocessing step.
    :param ground_truth: Ground truth labels
    :dataset_name: specific dataset name for clarity
    :compare_dim: Flag to check if changing the dimensions will affect accuracy. Note, if this flag is set to true, we 
                  take the 'dims' array and slice it so we only look at the first 9 elements. Use is the same for tSNE and UMAP
    :compare_neighbors: Flag to check if changign the number of neighbors will affect accuracy. Only look at elements starting from index 1 
                        of the 'dims' array. In tSNE, the 'dims' elements will be passed in as the argument for perplexity,
                        and in UMAP, the elements will be passed in as number of neighbors.

    :tSNE: Flag to check if tSNE or UMAP should be intialized,,
    :return: None

    """
   #Problem: instead of adding the title and the x y labels, it prints out a bunch of empty graphs 
   # and then a final graph with no labels but he actual points and lines are on it. Fix that + optimize this code
    tsne_ari = []
    tsne_aligned_acc = []
    
    umap_ari = []
    umap_aligned_acc = []
    
    pca_ari = []
    pca_aligned_acc = []
    dims = [1,2,3,4,5,6,10,30,50,100,150,200]
    if (compare_dim):
      #Neighbors at 30 for default val
      for i in range(len(dims) - 3):
        d_plot = visualize_umap(data, ground_truth, n_neighbors = 30, n_components = dims[i])
        k_means_umap_ari, k_means_umap_labels = k_means_with_umap(d_plot,ground_truth)
        k_means_pca_ari, k_means_pca_labels = k_means_with_pca(data,dims[i], ground_truth)

          umap_ari.append(k_means_umap_ari)
          pca_ari.append(k_means_pca_ari)

          umap_acc = calculate_aligned_accuracy(ground_truth, k_means_umap_labels)
          pca_acc = calculate_aligned_accuracy(ground_truth, k_means_pca_labels)

          umap_aligned_acc.append(umap_acc)
          pca_aligned_acc.append(pca_acc)

      dims = dims[:9]
      plt.title('Adjusted Rand Index (ARI) vs. Embedding Dimension for ' + dataset_name)
      plt.xlabel('Embedding Dimension')

      plt.figure(figsize=(12, 5))
      plt.subplot(1, 2, 1)
      plt.plot(dims, umap_ari, label='UMAP ARI', marker='o')
      plt.plot(dims, pca_ari, label='PCA ARI', marker='o')

      plt.ylabel('ARI')
      plt.legend()

      plt.subplot(1, 2, 2)
      plt.plot(dims, umap_aligned_acc, label='UMAP Accuracy', marker='o')
      plt.plot(dims, pca_aligned_acc, label='PCA Accuracy', marker='o')
      plt.ylabel('Accuracy')
      plt.legend()
      plt.tight_layout()
      plt.show()
    elif (compare_neighbors):
        for i in range(1,len(dims)):
          d_plot = visualize_umap(data,ground_truth,n_neighbors=dims[i],n_components=3)
          k_means_umap_ari, k_means_umap_labels = k_means_with_umap(d_plot,ground_truth)
          k_means_pca_ari, k_means_pca_labels = k_means_with_pca(data,3, ground_truth)

            umap_ari.append(k_means_umap_ari)
            pca_ari.append(k_means_pca_ari)

            umap_acc = calculate_aligned_accuracy(ground_truth, k_means_umap_labels)
            pca_acc = calculate_aligned_accuracy(ground_truth, k_means_pca_labels)

          umap_aligned_acc.append(umap_acc)
          pca_aligned_acc.append(pca_acc)
        dims = dims[1:]
      plt.title('Adjusted Rand Index (ARI) vs. Number of Neighbors')
      plt.xlabel('Number of Neighbors')
      plt.figure(figsize=(12, 5))
      plt.subplot(1, 2, 1)
      plt.plot(dims, umap_ari, label='UMAP ARI', marker='o')
      plt.plot(dims, pca_ari, label='PCA ARI', marker='o')

      plt.ylabel('ARI')
      plt.legend()

      plt.subplot(1, 2, 2)
      plt.plot(dims, umap_aligned_acc, label='UMAP Accuracy', marker='o')
      plt.plot(dims, pca_aligned_acc, label='PCA Accuracy', marker='o')
      plt.ylabel('Accuracy')
      plt.legend()
      plt.tight_layout()
      plt.show()
