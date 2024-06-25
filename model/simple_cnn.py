import torch
import torch.nn as nn
from utils import loadHSI
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset


import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(256 * 10 * 10, 1024)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 83*86)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, feature_extraction=False):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        if feature_extraction:
            return x
        
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.dropout2(x)
        x = self.softmax(x)
        return x


def calculate_aligned_accuracy(ground_truth, cluster_labels):
    true_labels = ground_truth.flatten()
    cm = confusion_matrix(true_labels, cluster_labels)

    row_ind, col_ind = linear_sum_assignment(-cm)
    label_mapping = {cluster_label: true_label for cluster_label, true_label in zip(col_ind, row_ind)}

    # Ensure all cluster labels are in the mapping, if not map to a dummy label
    max_true_label = max(true_labels)
    aligned_labels = []
    for label in cluster_labels:
        if label in label_mapping:
            aligned_labels.append(label_mapping[label])
        else:
            aligned_labels.append(max_true_label + 1)  # Mapping to a dummy label

    accuracy = np.mean(np.array(aligned_labels) == np.array(true_labels))

    return accuracy

def main():      
    salinasA_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_corrected.mat'
    salinasA_gt_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_gt.mat'
    
    # Load HSI data
    X, M, N, D, HSI, GT, Y, n, K = loadHSI(salinasA_path, salinasA_gt_path, 'salinasA_corrected', 'salinasA_gt')
    
    # Reshape HSI data
    HSI = X.reshape((M, N, D))  # Should already be (83, 86, 204)
    
    HSI = torch.from_numpy(HSI).float().permute(2, 0, 1).unsqueeze(1)  # Shape: (1, 204, 83, 86)
    labels = torch.from_numpy(GT).float().repeat(204, 1, 1)  # Shape: (204, 83, 86)

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
            
            labels_flat = labels.view(16, 83 * 86)

            loss = criterion(outputs, labels_flat)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

    model.eval()
    with torch.no_grad():
        features = []
    for inputs, in dataloader:
        feature = model(inputs, feature_extraction=True)
        features.append(feature.squeeze().numpy())
    features = np.array(features)
    print(f"Extracted features shape: {features.shape}")  # Should be (204, 256*10*10)

# You can now use these featur
    # patch_size = 32

    # model = SimpleCNN()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = nn.optim.Adam(model.parameters(), lr=0.001)
    # num_epochs = 20
    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0
    #     for inputs, labels in train_loader:
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    #     print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    # model.eval()
    # with torch.no_grad():
    #     features = []
    #     patch_labels = []
    #     for i in range(0, HSI.shape[2] - patch_size + 1, patch_size):
    #         for j in range(0, HSI.shape[3] - patch_size + 1, patch_size):
    #             patch = HSI[:, :, i:i + patch_size, j:j + patch_size]
    #             patch_features = model(patch, feature_extraction=True)
    #             features.append(patch_features.squeeze().numpy())
                
    #             # Extract the corresponding ground truth label for the patch
    #             label_patch = GT[i:i + patch_size, j:j + patch_size]
    #             patch_label = np.bincount(label_patch.flatten()).argmax()
    #             patch_labels.append(patch_label)

    #     features = np.array(features)
    #     patch_labels = np.array(patch_labels)        
    #     print(f"Number of patches: {len(features)}")
    #     print(f"Feature shape before reshaping: {features.shape}")

    #     features = features.reshape(-1, 4096)
    # unique_labels_count = np.unique(GT.flatten()).size
    # print("Number of unique labels in ground truth:", unique_labels_count)
    # print("Unique labels in ground truth:", np.unique(GT.flatten()))


    # knn = KNeighborsClassifier(n_neighbors=3)
    # knn.fit(features, patch_labels)
    # cluster_labels = knn.predict(features)
    # print("Unique predicted labels by KNN:", np.unique(cluster_labels))
    # accuracy = calculate_aligned_accuracy(patch_labels, cluster_labels)
    # print("Aligned Accuracy:", accuracy)



    
if __name__ == "__main__":
    main()
