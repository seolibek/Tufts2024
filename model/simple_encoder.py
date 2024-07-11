import torch
import torch.nn as nn
from utils import loadHSI, calculate_aligned_accuracy
from sklearn.cluster import KMeans
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll.base import scope

class SimpleAutoencoder(nn.Module):
    '''
    Simple Autoencoder to extract most relevant features from hyperspectral image.
    :param num_channels: Number of channels, loaded in as 'D' from loadHSI function
    
    '''
    def __init__(self,num_channels):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(32, 7),
            nn.Sigmoid()
            )
        self.decoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.Sigmoid(),
            nn.Unflatten(1, (32, 1, 1)),
            nn.ConvTranspose2d(32, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, num_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self._initialize_weights(mode = 'xavier')

    def _initialize_weights(self, mode):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if mode == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif mode == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if mode == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif mode == 'xavier':
                    nn.init.xavier_normal_(m.weight)
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

# def reassemble_image(patches, M, N, patch_size):
#     reconstructed_image = np.zeros((M, N, patches.shape[1]))  
#     idx = 0
#     for i in range(M):
#         for j in range(N):
#             patch = patches[idx].transpose(1, 2, 0) 
#             reconstructed_image[i, j, :] = patch  
#             idx += 1
#     return reconstructed_image


# def visualize_bands(original_image, reconstructed_image, num_bands=204):
#     fig, axs = plt.subplots(2, num_bands, figsize=(num_bands * 2, 4))
#     for i in range(num_bands):
#         orig_band = original_image[:, :, i]
#         recon_band = reconstructed_image[:, :, i]
#         orig_band = (orig_band - orig_band.min()) / (orig_band.max() - orig_band.min())
#         recon_band = (recon_band - recon_band.min()) / (recon_band.max() - recon_band.min())
#         axs[0, i].imshow(orig_band, cmap='gray')
#         axs[0, i].axis('off')
#         axs[1, i].imshow(recon_band, cmap='gray')
#         axs[1, i].axis('off')
#     plt.show()

# def visualize_intermediate(original, decoded, num_samples=2, num_bands=5, num_pixels=14):
#     print("Original shape:", original.shape)
#     print("Decoded shape:", decoded.shape)

#     num_bands = min(num_bands, original.shape[1])
#     num_samples = min(num_samples, original.shape[0])

#     fig, axs = plt.subplots(num_bands, num_samples * 2, figsize=(num_samples * 5, num_bands * 5))
#     for i in range(num_samples):
#         for j in range(num_bands):
#             orig_band = original[i, j, :, :].reshape(1, 1)
#             dec_band = decoded[i, j, :, :].reshape(1, 1)

#             orig_band_normalized = (orig_band - orig_band.min()) / (orig_band.max() - orig_band.min() + 1e-8)
#             dec_band_normalized = (dec_band - dec_band.min()) / (dec_band.max() - dec_band.min() + 1e-8)

#             print(f"Original Band {j+1} Sample {i+1} min-max:", orig_band.min(), orig_band.max())
#             print(f"Decoded Band {j+1} Sample {i+1} min-max:", dec_band.min(), dec_band.max())

#             axs[j, i * 2].imshow(orig_band_normalized, cmap='gray', aspect='auto')
#             axs[j, i * 2].set_title(f"Original Band {j+1}")
#             axs[j, i * 2].axis('off')
#             axs[j, i * 2 + 1].imshow(dec_band_normalized, cmap='gray', aspect='auto')
#             axs[j, i * 2 + 1].set_title(f"Reconstructed Band {j+1}")
#             axs[j, i * 2 + 1].axis('off')
#     plt.tight_layout()
#     plt.show()

# def visualize_encoded(encoded):
#     encoded_2d = TSNE(n_components=2).fit_transform(encoded)
#     plt.figure(figsize=(8, 8))
#     plt.scatter(encoded_2d[:, 0], encoded_2d[:, 1], s=5)
#     plt.title('t-SNE visualization of encoded features')
#     plt.show()

def preprocess_HSI(data_path,gt_path, data_name, gt_name, is_Salinas = False):
    '''
    Preprocessing HSI data for our needs. is_salinas flag sets the ground truth to 0 based indexing. All other datasets do not need 0 based.
    Loads in the dataset, scales the data for normalization, and reshapes whole image to collection of pixels.

    '''
    X, M, N, D, HSI, GT, Y, n, K = loadHSI(data_path, gt_path, data_name, gt_name)
    # if is_Salinas:
    #     
    GT = GT - 1
    HSI = X.reshape((M, N, D))  

    scaler = MinMaxScaler()
    HSI = scaler.fit_transform(HSI.reshape(-1, D)).reshape(M, N, D)
    pixels = HSI.reshape(-1, D, 1, 1)
    pixels = torch.from_numpy(pixels).float()
    return HSI, GT, M, N, D, pixels

def create_dataloader(pixels, batch_size = 256):
    '''Loads in data into tensor dataset. Best batch size hyperparam 256'''
    dataset = TensorDataset(pixels)
    dataloader = DataLoader(dataset, batch_size, shuffle=False)
    return dataloader

def train_autoencoder(model, dataloader, criterion, optimizer, num_epochs=25):
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
        print("Epoch %d, Loss: %f" % (epoch+1, running_loss/len(dataloader)))
    return model

def evaluate_autoencoder(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    reconstructed_pixels = []
    encoded_features = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0]
            reconstructed, encoded = model(inputs)
            loss = criterion(reconstructed, inputs)
            total_loss += loss.item()
            reconstructed_pixels.append(reconstructed.cpu().numpy())
            encoded_features.append(encoded.cpu().numpy())

    average_loss = total_loss / len(dataloader)
    print("Average Reconstruction Loss: %f" % average_loss)

    encoded_features = np.concatenate(encoded_features, axis=0).reshape(-1, 7)
    return encoded_features

def cluster_and_evaluate(encoded_features, GT, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(encoded_features)
    kmeans_labels = kmeans.labels_
    unique_clusters = np.unique(kmeans_labels)
    print("KMeans Clustering Performance:")
    print("Number of unique clusters: %f" % len(unique_clusters))
    print("Unique clusters: %f" % unique_clusters)
    kmeans_accuracy = calculate_aligned_accuracy(GT.flatten(), kmeans_labels)
    print("Aligned Accuracy: %f" % kmeans_accuracy)

def train_model(params):      
    datasets = {
        'A': ('/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/Salinas_corrected.mat', '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/Salinas_gt.mat', 'salinas_corrected', 'salinas_gt'),
        'B': ('/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_corrected.mat', '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_gt.mat', 'salinasA_corrected', 'salinasA_gt'),
        'C': ('/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/Indian_pines_corrected.mat', '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/Indian_pines_gt.mat', 'indian_pines_corrected', 'indian_pines_gt'),
        'D': ('/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/PaviaU.mat', '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/PaviaU_gt.mat', 'paviaU', 'paviaU_gt')
    }

    choice = input("Choose a dataset:\nA) Salinas\nB) SalinasA\nC) Indian Pines\nD) Pavia\nEnter your choice: ").strip().upper()
    
    if choice not in datasets:
        print("Invalid choice. Please choose A, B, C, or D.")
        return

    data_path, gt_path, data_name, gt_name = datasets[choice]
    
    HSI, GT, M, N, D, pixels = preprocess_HSI(data_path,gt_path,data_name,gt_name,is_Salinas=False) #get rid of issalinas flag  useless

    num_unique_clusters = len(np.unique(GT))
    dataloader = create_dataloader(pixels,batch_size=256) #play w this num later
    
    model = SimpleAutoencoder(num_channels=D)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001253003, weight_decay=1e-5)
    model = train_autoencoder(model,dataloader,criterion,optimizer)
    encoded_features = evaluate_autoencoder(model,dataloader,criterion)
    cluster_and_evaluate(encoded_features,GT,num_unique_clusters)
    print(f"Number of parameters: {count_parameters(model)}")


def main():
    space = {
        'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-2)),
        'hidden_dim': scope.int(hp.quniform('hidden_dim', 16, 64, 16)),
        'init_mode': hp.choice('init_mode', ['kaiming', 'xavier'])
    }

    trials = Trials()
    best = fmin(fn=train_model,
                space=space,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials)

    print("Best parameters: %s" % best)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    main()
