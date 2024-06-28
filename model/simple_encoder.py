import torch
import torch.nn as nn
from utils import loadHSI
from sklearn.cluster import KMeans
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import os

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=204, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.encoder_fc = nn.Linear(32 * 1 * 1, 128)

        # Decoder
        self.decoder_fc = nn.Linear(128, 32 * 1 * 1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=204, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_fc(x)
        encoded_features = self.relu(x)

        # Decoder
        x = self.decoder_fc(encoded_features)
        x = self.relu(x)
        x = x.view(x.size(0), 32, 1, 1)  
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)

        return x, encoded_features

def save_original_hsi_as_image(hsi, save_path):
    
    hsi_min = hsi.min(axis=(0, 1), keepdims=True)
    hsi_max = hsi.max(axis=(0, 1), keepdims=True)
    hsi_normalized = (hsi - hsi_min) / (hsi_max - hsi_min)

    # Take the first three channels to create an RGB image
    rgb_image = hsi_normalized[:, :, :3]
    rgb_image = (rgb_image * 255).astype(np.uint8)  # Convert to uint8

    # Create a PIL image from the numpy array
    img = Image.fromarray(rgb_image, mode='RGB')

    # Create directories if necessary
    directory = os.path.dirname(save_path)
    os.makedirs(directory, exist_ok=True)

    # Save the image
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
    # Initialize the reconstructed image and the patch count matrices
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

    # Normalize the reconstructed image by the number of patches contributing to each pixel
    reconstructed_image /= np.maximum(patch_count, 1)  # Avoid division by zero

    return reconstructed_image


def tensor_to_image(tensor):
    tensor = tensor.squeeze()  # Remove any singleton dimensions
    
    # Debugging statement to check tensor shape and values
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor min: {tensor.min()}, Tensor max: {tensor.max()}")
    
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0, 1]
    tensor = (tensor * 255).to(torch.uint8)  # Convert to uint8
    
    tensor = tensor.cpu().numpy()

    # Debugging statement to check converted tensor values
    print(f"Converted tensor min: {tensor.min()}, Converted tensor max: {tensor.max()}")
    
    # Ensure the tensor is in a format suitable for PIL
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



def main():      
    salinasA_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_corrected.mat'
    salinasA_gt_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_gt.mat'
    X, M, N, D, HSI, GT, Y, n, K = loadHSI(salinasA_path, salinasA_gt_path, 'salinasA_corrected', 'salinasA_gt')
    
    GT = GT - 1  # Convert to 0-based indexing.. necessary unfortunately whatever
    HSI = X.reshape((M, N, D))  
    save_path = "/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/reconstructed/orginal_img.png"  # Full file path including file name and extension
    directory = os.path.dirname(save_path)
    
    os.makedirs(directory, exist_ok=True)  # Create directories if they don't exist
    
    if os.path.isdir(save_path):
        raise IsADirectoryError(f"Save path '{save_path}' is a directory. Please provide a valid file path.")
    img = save_original_hsi_as_image(HSI,save_path)




    patch_size = 1
    patches = extract_patches(HSI, patch_size)
    patches = patches.reshape(-1, patch_size, patch_size, D)
    patches = torch.from_numpy(patches).float().permute(0, 3, 1, 2)  # Shape: (num_patches, D, patch_size, patch_size)
    
    labels = torch.from_numpy(GT).long().flatten()  # Shape: (M*N,)
    dataset = TensorDataset(patches)
    #dataset size is 7138, as expected
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleAutoencoder()
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
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

    
    model.eval()
    total_loss = 0.0
    reconstructed_patches = []

    # save_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/reconstructed'
    # os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
    
        for batch in dataloader:
            inputs = batch[0]  
            reconstructed, _ = model(inputs)
            loss = criterion(reconstructed, inputs)
            total_loss += loss.item()
            reconstructed_patches.append(reconstructed.cpu().numpy())
            
        average_loss = total_loss / len(dataloader)
        print(f"Average Reconstruction Loss: {average_loss}")
        reconstructed_patches = np.concatenate(reconstructed_patches, axis = 0)
        reconstructed_image = reassemble_image(reconstructed_patches, M, N, patch_size)
        img = tensor_to_image(torch.tensor(reconstructed_image))

        save_path = "/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/reconstructed/img.png"  # Full file path including file name and extension
        directory = os.path.dirname(save_path)
        
        os.makedirs(directory, exist_ok=True)  # Create directories if they don't exist
        
        if os.path.isdir(save_path):
            raise IsADirectoryError(f"Save path '{save_path}' is a directory. Please provide a valid file path.")
        img.save(save_path)
            
    # kmeans = KMeans(n_clusters=7, random_state=0).fit(features)
    # cluster_labels = kmeans.labels_
    # print(cluster_labels.shape) #shape is 7138,
    # print(GT.shape)

    # accuracy = calculate_aligned_accuracy(GT, cluster_labels)
    # print(f"Aligned Accuracy: {accuracy}")



    
if __name__ == "__main__":
    main()
