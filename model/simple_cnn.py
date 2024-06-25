import torch
import torch.nn as nn
from utils import loadHSI
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset


import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=204, out_channels=164, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=164, out_channels=100, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=100, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64, 128)  # Adjust the output size as needed
        self.fc2 = nn.Linear(128, 7)  


    def forward(self, x, feature_extraction = False):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)              

        x = self.conv2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        if feature_extraction:
            return x
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def calculate_aligned_accuracy(ground_truth, cluster_labels):
    true_labels = ground_truth.flatten()
    
    # assert len(true_labels) == len(cluster_labels), "Length of true_labels and cluster_labels must be the same"

    print(f"True labels shape: {true_labels.shape}")
    print(f"Cluster labels shape: {cluster_labels.shape}")
    print(f"Unique true labels: {np.unique(true_labels)}")
    print(f"Unique cluster labels: {np.unique(cluster_labels)}")

    cm = confusion_matrix(true_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    label_mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    aligned_labels = np.array([label_mapping[label] for label in cluster_labels])

    # Calculate accuracy
    accuracy = np.mean(aligned_labels == true_labels)

    return accuracy


def main():      
    salinasA_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_corrected.mat'
    salinasA_gt_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_gt.mat'
    
    # Load HSI data
    X, M, N, D, HSI, GT, Y, n, K = loadHSI(salinasA_path, salinasA_gt_path, 'salinasA_corrected', 'salinasA_gt')
    
    #adjusting ground truth 
    print("Original ground truth labels (unique values):", np.unique(GT))

    GT = GT - 1  # Convert to 0-based indexing
    print("Converted ground truth labels (unique values):", np.unique(GT))

    # Reshape HSI data
    HSI = X.reshape((M, N, D))  # Should already be (83, 86, 204)
    HSI = torch.from_numpy(HSI).float().permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 204, 83, 86)
    labels = torch.from_numpy(GT).long().flatten()  # Shape: (83*86,)

    print(HSI.shape)  # (1, 204, 83, 86)
    print(labels.shape)  # (83*86,)


    
    # Each pixel with all bands as channels
    HSI = HSI.permute(2, 3, 1, 0).reshape(-1, 204, 1, 1)  # Shape: (83*86, 204, 1, 1)
    labels = labels  # Shape: (83*86,)
    dataset = TensorDataset(HSI, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
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
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

    model.eval()
    with torch.no_grad():
        features = []
        for inputs, labels in dataloader:
            feature = model(inputs, feature_extraction=True)
            features.append(feature.cpu().numpy())
    features = np.vstack(features)
    print(f"Extracted features shape: {features.shape}")
    # print(f"Labels shape: {all_labels.shape}")

    kmeans = KMeans(n_clusters=7, random_state=0).fit(features)
    cluster_labels = kmeans.labels_
    print(cluster_labels.shape)
    print(GT.shape)

    # Calculate and print the aligned accuracy
    accuracy = calculate_aligned_accuracy(GT, cluster_labels)
    print(f"Aligned Accuracy: {accuracy}")



    
if __name__ == "__main__":
    main()
