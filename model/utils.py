import numpy as np
import scipy
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

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



def calculate_aligned_accuracy(ground_truth, cluster_labels):
    true_labels = ground_truth.flatten()
    cm = confusion_matrix(true_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    label_mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    aligned_labels = np.array([label_mapping[label] for label in cluster_labels])

    accuracy = np.mean(aligned_labels == true_labels)

    return accuracy