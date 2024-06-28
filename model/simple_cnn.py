import torch
import torch.nn as nn
from utils import loadHSI
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=204, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=100, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32*5*5, 128) #make this automized.. out channels * patch area
        # self.dropout = nn.Dropout(p=0.5)  
        self.fc2 = nn.Linear(128, 7)  


    def forward(self, x, feature_extraction = False):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)  

        if feature_extraction:
            return x
        # print(f'the shape here is hfkjsdfhsk{x.shape}') #(16,800)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x
    
def calculate_aligned_accuracy(ground_truth, cluster_labels):
    true_labels = ground_truth.flatten()
    cm = confusion_matrix(true_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    label_mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    aligned_labels = np.array([label_mapping[label] for label in cluster_labels])

    accuracy = np.mean(aligned_labels == true_labels)

    return accuracy

def extract_patches(hsi, patch_size):
    M, N, D = hsi.shape
    padded_hsi = np.pad(hsi, ((patch_size//2, patch_size//2), (patch_size//2, patch_size//2), (0, 0)), mode='reflect')
    patches = []
    for i in range(M):
        for j in range(N):
            patch = padded_hsi[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
    return np.array(patches)

def main():      
    salinasA_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_corrected.mat'
    salinasA_gt_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_gt.mat'
    X, M, N, D, HSI, GT, Y, n, K = loadHSI(salinasA_path, salinasA_gt_path, 'salinasA_corrected', 'salinasA_gt')
    
    GT = GT - 1  # Convert to 0-based indexing.. necessary unfortunately whatever
    # Reshape HSI data
    HSI = X.reshape((M, N, D))  
    patch_size = 5
    patches = extract_patches(HSI, patch_size)
    patches = patches.reshape(-1, patch_size, patch_size, D)
    patches = torch.from_numpy(patches).float().permute(0, 3, 1, 2)  # Shape: (num_patches, D, patch_size, patch_size)
    labels = torch.from_numpy(GT).long().flatten()  # Shape: (M*N,)

    dataset = TensorDataset(patches, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)
   
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

    model.eval()
    with torch.no_grad():
        features = []
        for inputs, labels in dataloader:
            feature = model(inputs, feature_extraction=True)
            features.append(feature.cpu().numpy())
    features = np.vstack(features)

    print(f"Extracted features shape: {features.shape}")

    kmeans = KMeans(n_clusters=7, random_state=0).fit(features)
    cluster_labels = kmeans.labels_
    print(cluster_labels.shape)
    print(GT.shape)

    accuracy = calculate_aligned_accuracy(GT, cluster_labels)
    print(f"Aligned Accuracy: {accuracy}")



    
if __name__ == "__main__":
    main()
