import torch
import torch.nn as nn
from utils import loadHSI, calculate_aligned_accuracy
from sklearn.cluster import KMeans
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
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(32, 7),
            nn.BatchNorm1d(7),  # Use BatchNorm1d for linear layers
            nn.LeakyReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.BatchNorm1d(32),  # Use BatchNorm1d for linear layers
            nn.LeakyReLU(),
            nn.Unflatten(1, (32, 1, 1)),
            nn.ConvTranspose2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 204, kernel_size=1),
            nn.BatchNorm2d(204),
            nn.LeakyReLU()
        )
        # # Encoder
        # self.conv1 = nn.Conv2d(in_channels=204, out_channels=128, kernel_size=1, stride=1, padding=0)
        # self.bn1 = nn.BatchNorm2d(128)
        # self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.encoder_fc1 = nn.Linear(32, 7)
        # self.relu = nn.LeakyReLU()

        # # Decoder
        # self.decoder_fc1 = nn.Linear(7, 32)
        # self.bn_fc6 = nn.BatchNorm1d(32)
        # self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        # self.bn5 = nn.BatchNorm2d(128)
        # self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=204, kernel_size=1, stride=1, padding=0)
        # self.bn6 = nn.BatchNorm2d(204)
        
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
        print(f'input shape is {x.shape}')

        encoded = self.encoder(x)
        print(f'encoded shape shape is {encoded.shape}')

        decoded = self.decoder(encoded)
        print(f'decoded shape is {decoded.shape}')
        return decoded, encoded

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
    reconstructed_image = np.zeros((M, N, patches.shape[1]))  # Adjusted to use patches.shape[1] for the channel dimension
    idx = 0
    for i in range(M):
        for j in range(N):
            patch = patches[idx].transpose(1, 2, 0)  # Match visualization function
            patch = (patch - patch.min()) / (patch.max() - patch.min())  # Normalize each patch
            reconstructed_image[i, j, :] = patch  # Directly assign the patch values to the corresponding pixel
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


def visualize_reconstructed_patches(patches, num_patches=25):
    fig, axs = plt.subplots(1, num_patches, figsize=(15, 5))
    for i in range(num_patches):
        patch = patches[i].transpose(1, 2, 0)
        patch = (patch - patch.min()) / (patch.max() - patch.min())  # Normalize to [0, 1]
        axs[i].imshow(patch[:, :, :3])
        axs[i].axis('off')
    plt.show()

def visualize_reassembled_image(image):
    fig, ax = plt.subplots(figsize=(10, 10))
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    rgb_image = image[:, :, :3]
    ax.imshow(rgb_image)
    ax.axis('off')
    plt.show()

def main():      
    salinasA_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_corrected.mat'
    salinasA_gt_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_gt.mat'
    X, M, N, D, HSI, GT, Y, n, K = loadHSI(salinasA_path, salinasA_gt_path, 'salinasA_corrected', 'salinasA_gt')
    print(f"Original labels sample: {GT[:10]}")  # Print first 10 labels

    
    GT = GT - 1  # Convert to 0-based indexing.. necessary
    print(f"Converted labels sample: {GT[:10]}")  # Print first 10 converted labels

    HSI = X.reshape((M, N, D))  
    save_path = "/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/reconstructed/orginal_img.png" 
    directory = os.path.dirname(save_path)
    
    os.makedirs(directory, exist_ok=True)  
    
    if os.path.isdir(save_path):
        raise IsADirectoryError(f"Save path '{save_path}' is a directory. Please provide a valid file path.")
    img = save_original_hsi_as_image(HSI,save_path)

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
    
    num_epochs = 15
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
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

    
    model.eval()
    total_loss = 0.0
    reconstructed_patches = []
    feature_list = []

    # save_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/reconstructed'
    # os.makedirs(save_path, exist_ok=True)
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
    print(f"Sample reconstructed patches: {reconstructed_patches[:5]}")
    reconstructed_patches = np.concatenate(reconstructed_patches, axis = 0)
    visualize_reconstructed_patches(reconstructed_patches)

    reconstructed_image = reassemble_image(reconstructed_patches, M, N, patch_size)
    
    visualize_reassembled_image(reconstructed_image)

    
    feature_list = np.vstack(feature_list) 

    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_list)

    kmeans = KMeans(n_clusters=7, random_state=0).fit(normalized_features)
    cluster_labels = kmeans.labels_




    unique_clusters = np.unique(cluster_labels)
    print(f"Number of unique clusters: {len(unique_clusters)}")
    print(f"Unique clusters: {unique_clusters}")

    accuracy = calculate_aligned_accuracy(GT, cluster_labels)
    print(f"Aligned Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
