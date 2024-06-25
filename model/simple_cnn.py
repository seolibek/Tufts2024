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
    def __init__(self, final_dims):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=204, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=final_dims, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(final_dims)
        
        self.fc1 = nn.Linear(final_dims * 83 * 86, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 83 * 86)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 83, 86)  # Reshape to (83, 86)
        return x


def calculate_aligned_accuracy(ground_truth, cluster_labels):
    true_labels = ground_truth.flatten()
    cm = confusion_matrix(true_labels, cluster_labels)

    row_ind, col_ind = linear_sum_assignment(-cm)
    label_mapping = {cluster_label: true_label for cluster_label, true_label in zip(col_ind, row_ind)}
    aligned_labels = [label_mapping[label] for label in cluster_labels]

    accuracy = np.mean(np.array(aligned_labels) == np.array(true_labels))

    return accuracy

def main():      
    salinasA_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_corrected.mat'
    salinasA_gt_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_gt.mat'
    
    # Load HSI data
    X, M, N, D, HSI, GT, Y, n, K = loadHSI(salinasA_path, salinasA_gt_path, 'salinasA_corrected', 'salinasA_gt')
    
    # Reshape HSI data
    HSI = X.reshape((M, N, D))  # Should already be (83, 86, 204)
    
    HSI = torch.from_numpy(HSI).float().permute(2, 0, 1)  # Shape: (1, 204, 83, 86)
    final_dims = 64
    model = SimpleCNN(final_dims) #can adjust to change dims????

    with torch.no_grad():
        reduced_representation = model(HSI)
    print(reduced_representation.shape)

    reduced_representation = reduced_representation.squeeze().permute(1, 2, 0).cpu().numpy()  # Shape (83, 86, 32)
    reduced_representation_flat = reduced_representation.reshape(-1, final_dims)  # Shape (83*86, final_dims)

    kmeans = KMeans(n_clusters=7)  
    kmeans.fit(reduced_representation_flat) 

    # Get the cluster labels
    cluster_labels = kmeans.labels_ 

    print(cluster_labels.shape)
    cluster_labels_reshaped = cluster_labels.reshape(83, 86)
    GT = GT.flatten()
    cluster_labels_reshaped = cluster_labels_reshaped.flatten()
    print(cluster_labels_reshaped)
    accuracy = calculate_aligned_accuracy(GT,cluster_labels_reshaped)
    print(accuracy)




#     labels = torch.from_numpy(GT).float().repeat(204, 1, 1)  # Shape: (204, 83, 86)

#     dataset = TensorDataset(HSI, labels)
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

#     model = SimpleCNN()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     num_epochs = 20
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, labels in dataloader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
            
#             labels_flat = labels.view(16, 83 * 86)

#             loss = criterion(outputs, labels_flat)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

#     model.eval()
#     with torch.no_grad():
#         features = []
#     for inputs, in dataloader:
#         feature = model(inputs, feature_extraction=True)
#         features.append(feature.squeeze().numpy())
#     features = np.array(features)
#     print(f"Extracted features shape: {features.shape}")  # Should be (204, 256*10*10)

#     # knn = KNeighborsClassifier(n_neighbors=3)
#     # knn.fit(features, patch_labels)
#     # cluster_labels = knn.predict(features)
#     # print("Unique predicted labels by KNN:", np.unique(cluster_labels))
#     # accuracy = calculate_aligned_accuracy(patch_labels, cluster_labels)
#     # print("Aligned Accuracy:", accuracy)



    
if __name__ == "__main__":
    main()
