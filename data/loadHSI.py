import numpy as np
import scipy.io

def load_hsi(hsi_name):
    if hsi_name == 'SalinasA':
        data = scipy.io.loadmat('SalinasA_corrected.mat')
        HSI = data['salinasA_corrected']
        data = scipy.io.loadmat('SalinasA_gt.mat')
        GT = data['salinasA_gt']

    elif hsi_name == 'IndianPines':
        data = scipy.io.loadmat('Indian_pines_corrected.mat')
        HSI = data['indian_pines_corrected']
        data = scipy.io.loadmat('indian_pines_gt.mat')
        GT = data['indian_pines_gt']

    elif hsi_name == 'Salinas':
        data = scipy.io.loadmat('Salinas_corrected.mat')
        HSI = data['salinas_corrected']
        data = scipy.io.loadmat('Salinas_gt.mat')
        GT = data['salinas_gt']

    elif hsi_name == 'WHU':
        data = scipy.io.loadmat('WHU_Hi_LongKou.mat')
        HSI = data['WHU_Hi_LongKou']
        data = scipy.io.loadmat('WHU_Hi_LongKou_gt.mat')
        GT = data['WHU_Hi_LongKou_gt']

    elif hsi_name == 'JasperRidge':
        data = scipy.io.loadmat('jasperRidge2_R198.mat')
        HSI = np.reshape(data['Y'].T, (data['nRow'][0][0], data['nCol'][0][0], len(data['SlectBands'][0])))
        y_data = scipy.io.loadmat('end4.mat')
        _, Y = np.max(y_data['A'].T, axis=1), y_data['A'].T.argmax(axis=1)
        GT = np.reshape(Y, (data['nRow'][0][0], data['nCol'][0][0]))

    elif hsi_name == 'Pavia Subset':
        data = scipy.io.loadmat('Pavia_gt.mat')
        GT = data['pavia_gt'][201:400, 430:530]
        data = scipy.io.loadmat('Pavia.mat')
        HSI = data['pavia'][201:400, 430:530, :]

    M, N, D = HSI.shape
    n = M * N
    X = HSI.reshape((n, D))

    #normalzie columns

    norms = np.sqrt(np.sum(X**2, axius = 0))
    X /= norms
    

    if hsi_name == 'SalinasA':
        X += 1e-6 * np.random.randn(*X.shape)

    HSI = X.reshape((M, N, D))

    # Correct GT labels
    new_gt = np.zeros_like(GT)
    unique_classes = np.unique(GT)
    K = len(unique_classes)
    for k, uc in enumerate(K, start = 1):
        new_gt[GT == uc] = k
    Y = new_gt.reshape((n, 1))
    GT = new_gt

    return X, M, N, D, HSI, GT, Y, n, K

