import torch
import torch.nn as nn
from umap_script import loadHSI
from torchvision import transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


#currently adjusted to salinas A, size 83 x 86, 204 bands
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # (B, 1, 64, 64) -> (B, 16, 32, 32)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (B, 16, 32, 32) -> (B, 32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 32, 16, 16) -> (B, 64, 8, 8)
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 64, 8, 8) -> (B, 32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, 16, 16) -> (B, 16, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # (B, 16, 32, 32) -> (B, 1, 64, 64)
            nn.Sigmoid()  # Using Sigmoid to get outputs between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class ResizeAndToTensor:
        def __init__(self, size):
            self.size = size

        def __call__(self, band, gt):

            band = torch.from_numpy(band).float()  
            gt = torch.from_numpy(gt).float()      

            band = band.unsqueeze(0).unsqueeze(0) #(83, 86) -> (1, 1, 83, 86)
            band = F.interpolate(band, size=self.size, mode='bilinear', align_corners=False) #(1, 1, 83, 86) -> (1, 1, 64, 64)
            band = band.squeeze(0) #(1, 1, 64, 64) -> (1, 64, 64) squeeze only once - keeping 1 for the channel dims, which  isnt needed for gt

            gt = gt.unsqueeze(0).unsqueeze(0)
            gt = F.interpolate(gt, size=self.size, mode='nearest')
            gt = gt.squeeze(0).squeeze(0) #(1, 1, 64, 64) -> (64, 64)

            return band, gt
class DataFormatter:
        def __init__(self):
            return None
        def __call__(self,hsi,gt):
            transform = ResizeAndToTensor(size=(64, 64))
            HSI_transformed, GT_transformed = transform(hsi, gt)

            HSI_transformed = []
            GT_transformed = []

            for band_idx in range(hsi.shape[2]):  # looping over each spectral band, treating them like samples
                band = hsi[:, :, band_idx]
                band_t, gt_t = transform(band, gt)
                HSI_transformed.append(band_t)
                GT_transformed.append(gt_t)

            # Convert lists to tensors.. idk y but not doing this breaks training process later...
            HSI_transformed = torch.stack(HSI_transformed)
            GT_transformed = torch.stack(GT_transformed)

            HSI_transformed = HSI_transformed.unsqueeze(1)  # reshaped for the num spectral bands (204, 1, 64, 64)
            HSI_train, HSI_test, GT_train, GT_test = train_test_split(HSI_transformed, GT_transformed, test_size=0.2, random_state=42)

            train_dataset = torch.utils.data.TensorDataset(HSI_train, GT_train)
            test_dataset = torch.utils.data.TensorDataset(HSI_test, GT_test)

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128)

            return train_loader,test_loader

def main():
    model = Autoencoder()

    #abstract out path later
    salinasA_path = 'data/SalinasA_corrected.mat'
    salinasA_gt_path = 'data/SalinasA_gt.mat'
    #and loadHSI stuff..

    X, M, N, D, HSI, GT, Y, n, K = loadHSI(salinasA_path, salinasA_gt_path, 'salinasA_corrected', 'salinasA_gt')
    #i am redefnig HSI here bc it became weired from some of the other stuff we did in load HSI where we were flattening the image

    #above, loaded data, where the data is HSI and ground truth is GT. then shape of the HSI data is 83 x 86 x 204 and GT is 83 x 86
    
    HSI = X.reshape((M, N, D)) 
    train_loader,test_loader = DataFormatter(HSI,GT)
    
    # sanity check,.
    # for hsi_batch, gt_batch in train_loader:
    #     print("HSI Batch Shape:", hsi_batch.shape)  # Should be (batch_size, 1, 64, 64)
    #     print("GT Batch Shape:", gt_batch.shape)    # Should be (batch_size, 64, 64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the autoencoder
    num_epochs = 15 #i just want it to stop training pretty fast, make it higher later

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (hsi_batch, gt_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(hsi_batch)
            loss = mse(outputs, hsi_batch) 
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
    torch.save(model.state_dict(), 'conv_autoencoder.pth')
