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
        self.encoder_fc = nn.Linear(32 * 5 * 5, 128)

        # Decoder
        self.decoder_fc = nn.Linear(128, 32 * 5 * 5)
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
        x = x.view(x.size(0), 32, 5, 5)  # Adjust the shape to match the output of the last conv layer in the encoder
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)

        return x, encoded_features

def extract_patches(hsi, patch_size):
    M, N, D = hsi.shape
    padded_hsi = np.pad(hsi, ((patch_size//2, patch_size//2), (patch_size//2, patch_size//2), (0, 0)), mode='reflect')
    patches = []
    for i in range(M):
        for j in range(N):
            patch = padded_hsi[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
    return np.array(patches)


def tensor_to_image(tensor):
    tensor = tensor.cpu().detach()
    
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    tensor = tensor[:3, :, :]  # Taking only the first 3 channels for visualization as RGB

    tensor = tensor.numpy()
    tensor = np.transpose(tensor, (1, 2, 0))  # Change to HWC format
    image = Image.fromarray((tensor * 255).astype('uint8'))
    return image



def main():      
    salinasA_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_corrected.mat'
    salinasA_gt_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_gt.mat'
    X, M, N, D, HSI, GT, Y, n, K = loadHSI(salinasA_path, salinasA_gt_path, 'salinasA_corrected', 'salinasA_gt')
    
    GT = GT - 1  # Convert to 0-based indexing.. necessary unfortunately whatever
    HSI = X.reshape((M, N, D))  
    patch_size = 5
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
    save_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/reconstructed'
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
    
        for i, batch in enumerate(dataloader):
            inputs = batch[0]  
            reconstructed, _ = model(inputs)
            loss = criterion(reconstructed, inputs)
            total_loss += loss.item()
            
            for j, rec in enumerate(reconstructed):
                img = tensor_to_image(rec)
                img.save(os.path.join(save_path, f'reconstructed_image_{i*inputs.size(0) + j}.png'))
        
        average_loss = total_loss / len(dataloader)
        print(f"Average Reconstruction Loss: {average_loss}")

    # kmeans = KMeans(n_clusters=7, random_state=0).fit(features)
    # cluster_labels = kmeans.labels_
    # print(cluster_labels.shape) #shape is 7138,
    # print(GT.shape)

    # accuracy = calculate_aligned_accuracy(GT, cluster_labels)
    # print(f"Aligned Accuracy: {accuracy}")



    
if __name__ == "__main__":
    main()
