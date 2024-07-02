import torch
import torch.nn as nn
from utils import loadHSI, calculate_aligned_accuracy
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(204, 128, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(32, 7),
            nn.LeakyReLU(),
            nn.BatchNorm1d(7),
            nn.Dropout(0.3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Unflatten(1, (32, 1, 1)),
            nn.ConvTranspose2d(32, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 128, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 204, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(204)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights to avoid extremely small values
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)
        return decoded, encoded

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
    reconstructed_image = np.zeros((M, N, patches.shape[1]))  
    idx = 0
    for i in range(M):
        for j in range(N):
            patch = patches[idx].transpose(1, 2, 0) 
            reconstructed_image[i, j, :] = patch  
            idx += 1
    return reconstructed_image


def tensor_to_image(tensor):
    tensor = tensor.squeeze()  
    
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) 
    tensor = (tensor * 255).to(torch.uint8)  
    
    tensor = tensor.cpu().numpy()
    
    if tensor.ndim == 2:  # Grayscale image
        return Image.fromarray(tensor, mode='L')
    elif tensor.ndim == 3 and tensor.shape[2] == 3:  # RGB image
        return Image.fromarray(tensor, mode='RGB')
    elif tensor.ndim == 3 and tensor.shape[2] == 204:  # Hyperspectral image, need to reduce dimensions
        # Example: Convert to RGB by taking the first three channels
        rgb_image = tensor[:, :, :3]
        return Image.fromarray(rgb_image, mode='RGB')
    else:
        raise ValueError("Unexpected tensor shape for image conversion")

def visualize_bands(original_image, reconstructed_image, num_bands=204):
    fig, axs = plt.subplots(2, num_bands, figsize=(num_bands * 2, 4))
    for i in range(num_bands):
        orig_band = original_image[:, :, i]
        recon_band = reconstructed_image[:, :, i]
        orig_band = (orig_band - orig_band.min()) / (orig_band.max() - orig_band.min())
        recon_band = (recon_band - recon_band.min()) / (recon_band.max() - recon_band.min())
        axs[0, i].imshow(orig_band, cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(recon_band, cmap='gray')
        axs[1, i].axis('off')
    plt.show()

def visualize_intermediate(encoded, decoded, original, num_samples=10):
    fig, axs = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
    for i in range(num_samples):
        orig_sample = original[i, :].reshape(1, -1)
        enc_sample = encoded[i, :].reshape(1, -1)
        dec_sample = decoded[i, :].reshape(1, -1)
        axs[0, i].imshow(orig_sample, cmap='gray', aspect='auto')
        axs[0, i].axis('off')
        axs[1, i].imshow(enc_sample, cmap='gray', aspect='auto')
        axs[1, i].axis('off')
        axs[2, i].imshow(dec_sample, cmap='gray', aspect='auto')
        axs[2, i].axis('off')
    plt.show()

def main():      
    salinasA_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_corrected.mat'
    salinasA_gt_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_gt.mat'
    X, M, N, D, HSI, GT, Y, n, K = loadHSI(salinasA_path, salinasA_gt_path, 'salinasA_corrected', 'salinasA_gt')

    
    GT = GT - 1  # Convert to 0-based indexing.. necessary

    HSI = X.reshape((M, N, D))  

    patch_size = 1
    patches = extract_patches(HSI, patch_size)
    print(f"Extracted patches shape: {patches.shape}") #(7138,1,1,204)

    patches = torch.from_numpy(patches).float().permute(0, 3, 1, 2)  # Shape: (num_patches, D, patch_size, patch_size) -> (7138,204,1,1)
    print(f"Patches permuted for input: {patches.shape}")

    labels = torch.from_numpy(GT).long().flatten()  # Shape: (M*N,)
    dataset = TensorDataset(patches)
    dataloader = DataLoader(dataset, batch_size=14, shuffle=True)
    model = SimpleAutoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs in dataloader:
            inputs = inputs[0]
            optimizer.zero_grad()
            reconstructed, encoded = model(inputs)
            loss = criterion(reconstructed, inputs)  
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

         # Visualize intermediate outputs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                inputs = next(iter(dataloader))[0]
                reconstructed, encoded = model(inputs)
                visualize_intermediate(encoded.view(-1, encoded.size(1)).cpu().numpy(), reconstructed.view(-1, reconstructed.size(1)).cpu().numpy(), inputs.view(-1, inputs.size(1)).cpu().numpy())


    
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

    reconstructed_patches = np.concatenate(reconstructed_patches, axis = 0)
    reconstructed_image = reassemble_image(reconstructed_patches, M, N, patch_size)
    visualize_bands(HSI, reconstructed_image)
    feature_list = np.vstack(feature_list) 

    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_list)

    kmeans = KMeans(n_clusters=7, random_state=0).fit(normalized_features)
    dbscan = DBSCAN(eps=0.5, min_samples=5).fit(normalized_features)
    agglo = AgglomerativeClustering(n_clusters=7).fit(normalized_features)

    # Evaluate clustering performance
    kmeans_labels = kmeans.labels_
    dbscan_labels = dbscan.labels_
    agglo_labels = agglo.labels_

    print("KMeans Clustering Performance:")
    kmeans_accuracy = calculate_aligned_accuracy(GT.flatten(), kmeans_labels)
    print(f"Aligned Accuracy: {kmeans_accuracy}")

    print("DBSCAN Clustering Performance:")
    dbscan_accuracy = calculate_aligned_accuracy(GT.flatten(), dbscan_labels)
    print(f"Aligned Accuracy: {dbscan_accuracy}")

    print("Agglomerative Clustering Performance:")
    agglo_accuracy = calculate_aligned_accuracy(GT.flatten(), agglo_labels)
    print(f"Aligned Accuracy: {agglo_accuracy}")

    reconstructed_pixels = np.concatenate(reconstructed_pixels, axis=0)
    reconstructed_image = reassemble_image(reconstructed_pixels.reshape(-1, D), M, N)
    reconstructed_image = scaler.inverse_transform(reconstructed_image.reshape(-1, D)).reshape(M, N, D)
    visualize_bands(HSI, reconstructed_image)


if __name__ == "__main__":
    main()
