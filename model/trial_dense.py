import torch
import torch.nn as nn
from utils import loadHSI, calculate_aligned_accuracy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import os

class FullyConnectedAutoencoder(nn.Module):
    def __init__(self):
        super(FullyConnectedAutoencoder, self).__init__()
        # Encoder
        self.encoder_fc1 = nn.Linear(204, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.encoder_fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.encoder_fc3 = nn.Linear(128, 64)
        self.bn_fc3 = nn.BatchNorm1d(64)
        self.encoder_fc4 = nn.Linear(64, 7)
        self.bn_fc4 = nn.BatchNorm1d(7)
        self.relu = nn.LeakyReLU()

        # Decoder
        self.decoder_fc1 = nn.Linear(7, 64)
        self.bn_fc5 = nn.BatchNorm1d(64)
        self.decoder_fc2 = nn.Linear(64, 128)
        self.bn_fc6 = nn.BatchNorm1d(128)
        self.decoder_fc3 = nn.Linear(128, 256)
        self.bn_fc7 = nn.BatchNorm1d(256)
        self.decoder_fc4 = nn.Linear(256, 204)
        self.bn_fc8 = nn.BatchNorm1d(204)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x = self.encoder_fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.encoder_fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)
        x = self.encoder_fc3(x)
        x = self.bn_fc3(x)
        x = self.relu(x)
        x = self.encoder_fc4(x)
        encoded_features = self.bn_fc4(x)
        encoded_features = self.relu(encoded_features)

        # Decoder
        x = self.decoder_fc1(encoded_features)
        x = self.bn_fc5(x)
        x = self.relu(x)
        x = self.decoder_fc2(x)
        x = self.bn_fc6(x)
        x = self.relu(x)
        x = self.decoder_fc3(x)
        x = self.bn_fc7(x)
        x = self.relu(x)
        x = self.decoder_fc4(x)
        x = self.bn_fc8(x)
        x = self.relu(x)

        return x, encoded_features

def save_original_hsi_as_image(hsi, save_path):
    hsi_min = hsi.min(axis=(0, 1), keepdims=True)
    hsi_max = hsi.max(axis=(0, 1), keepdims=True)
    hsi_normalized = (hsi - hsi_min) / (hsi_max - hsi_min)

    rgb_image = hsi_normalized[:, :, :3]
    rgb_image = (rgb_image * 255).astype(np.uint8)  # Convert to uint8

    img = Image.fromarray(rgb_image, mode='RGB')

    directory = os.path.dirname(save_path)
    os.makedirs(directory, exist_ok=True)

    img.save(save_path)
    print(f"Original HSI image saved as RGB at: {save_path}")

def extract_patches(hsi, patch_size):
    M, N, D = hsi.shape
    padded_hsi = np.pad(hsi, ((patch_size//2, patch_size//2), (patch_size//2, patch_size//2), (0, 0)), mode='reflect')
    patches = []
    for i in range(M):
        for j in range(N):
            patch = padded_hsi[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
    return np.array(patches)

def reassemble_image(patches, M, N, patch_size):
    reconstructed_image = np.zeros((M, N, 204))
    patch_count = np.zeros((M, N, 204))

    idx = 0
    for i in range(M):
        for j in range(N):
            patch = patches[idx].transpose(1, 2, 0)  # Transpose to (height, width, channels)
            i_start = i
            i_end = i + 1
            j_start = j
            j_end = j + 1
            reconstructed_image[i_start:i_end, j_start:j_end, :] += patch[patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1, :]
            patch_count[i_start:i_end, j_start:j_end, :] += 1
            idx += 1

    reconstructed_image /= np.maximum(patch_count, 1)

    return reconstructed_image

def tensor_to_image(tensor):
    tensor = tensor.squeeze()

    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor min: {tensor.min()}, Tensor max: {tensor.max()}")

    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = (tensor * 255).to(torch.uint8)

    tensor = tensor.cpu().numpy()

    if tensor.ndim == 2:  # Grayscale image
        return Image.fromarray(tensor, mode='L')
    elif tensor.ndim == 3 and tensor.shape[2] == 3:  # RGB image
        return Image.fromarray(tensor, mode='RGB')
    elif tensor.ndim == 3 and tensor.shape[2] == 204:  # Hyperspectral image, need to reduce dimensions
        rgb_image = tensor[:, :, :3]
        return Image.fromarray(rgb_image, mode='RGB')
    else:
        raise ValueError("Unexpected tensor shape for image conversion")

def main():
    salinasA_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_corrected.mat'
    salinasA_gt_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_gt.mat'
    X, M, N, D, HSI, GT, Y, n, K = loadHSI(salinasA_path, salinasA_gt_path, 'salinasA_corrected', 'salinasA_gt')

    GT = GT - 1  # Convert to 0-based indexing.. necessary
    HSI = X.reshape((M, N, D))
    save_path = "/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/reconstructed/orginal_img.png"
    directory = os.path.dirname(save_path)

    os.makedirs(directory, exist_ok=True)

    if os.path.isdir(save_path):
        raise IsADirectoryError(f"Save path '{save_path}' is a directory. Please provide a valid file path.")
    img = save_original_hsi_as_image(HSI, save_path)

    patch_size = 1
    patches = extract_patches(HSI, patch_size)
    print(f"Extracted patches shape: {patches.shape}")  # Expected: (7138, 1, 1, 204)

    patches = torch.from_numpy(patches).float().permute(0, 3, 1, 2)  # Shape: (7138, 204, 1, 1)
    patches = patches.view(-1, 204)  # Flatten to (7138, 204)
    print(f"Patches reshaped for input: {patches.shape}")

    labels = torch.from_numpy(GT).long().flatten()  # Shape: (M*N,)
    dataset = TensorDataset(patches)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    torch.set_printoptions(profile='full')

    model = FullyConnectedAutoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs in dataloader:
            inputs = inputs[0]
            optimizer.zero_grad()
            reconstructed, _ = model(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

    model.eval()
    total_loss = 0.0
    reconstructed_patches = []
    feature_list = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0]
            reconstructed, features = model(inputs)
            loss = criterion(reconstructed, inputs)
            total_loss += loss.item()
            reconstructed_patches.append(reconstructed.cpu().numpy())
            feature_list.append(features.cpu().numpy())

        average_loss = total_loss / len(dataloader)
        print(f"Average Reconstruction Loss: {average_loss}")
        # reconstructed_patches = np.concatenate(reconstructed_patches, axis=0)
        # reconstructed_image = reassemble_image(reconstructed_patches, M, N, patch_size)
        # img = tensor_to_image(torch.tensor(reconstructed_image))

        # save_path = "/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/reconstructed/img.png"
        # directory = os.path.dirname(save_path)

        # os.makedirs(directory, exist_ok=True)

        # if os.path.isdir(save_path):
        #     raise IsADirectoryError(f"Save path '{save_path}' is a directory. Please provide a valid file path.")
        # img.save(save_path)

    feature_list = np.vstack(feature_list)
    print(f"Extracted features shape: {feature_list.shape}")

    # Scale features by a large constant
    scaling_factor = 1e10
    scaled_features = feature_list * scaling_factor
    print(f"Scaled features sample: {scaled_features[:5]}")

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(scaled_features)
    print(f"Normalized features sample: {normalized_features[:5]}")

    # Check for unique features
    unique_features = np.unique(normalized_features, axis=0)
    print(f"Number of unique features: {unique_features.shape[0]}")
    print(f"Unique features: {unique_features}")

    # Fit KMeans and check cluster labels shape
    kmeans = KMeans(n_clusters=7, random_state=0).fit(normalized_features)
    cluster_labels = kmeans.labels_
    print(f"Cluster labels shape: {cluster_labels.shape}")  # Should be (7138,)
    print(f"GT shape: {GT.shape}")  # Should be (7138,)

    # Ensure the cluster labels match the ground truth shape
    cluster_labels = cluster_labels.reshape(M, N)
    print(f"Reshaped cluster labels shape: {cluster_labels.shape}")

    accuracy = calculate_aligned_accuracy(GT, cluster_labels.flatten())
    print(f"Aligned Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
