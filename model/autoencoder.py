import torch
import torch.nn as nn
from utils import loadHSI
from torchvision import transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
print("import complete")


#########################
# TODO:

# hsi is a single hyperspectral image with shape (h,w,c), where c represents number of spectral bands. and ground truth is the ground truth clustering
#  representation in (h,w) shape. I want to think of the 'c' as the number of samples, but because I am looking at the same photo and want to prevent 
# data leaks, I want to split up the original image into fourths, use 3/4 of it for training, and 1/4 for testing. So in my training step, each epoch 
# should still have 204 channels * 3 = 612 samples, and the testing step should have 204 samples. 



#currently adjusted to salinas A, size 83 x 86, 204 bands
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        print("autoencoder initialized")
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # (B, 1, H, W) -> (B, 16, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (B, 16, H/2, W/2) -> (B, 32, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 32, H/4, W/4) -> (B, 64, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 64, H/8, W/8) -> (B, 128, H/16, W/16)
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 128, H/16, W/16) -> (B, 64, H/8, W/8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 64, H/8, W/8) -> (B, 32, H/4, W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, H/4, W/4) -> (B, 16, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 16, H/2, W/2) -> (B, 1, H, W)
            nn.Sigmoid()  
        )

    def forward(self, x):
        self.encoded = self.encoder(x)
        x = self.decoder(self.encoded)
        return x
    def embedding(self):
        return self.encoded

class DataFormatter:
        def __init__(self, hsi):
            self.hsi = hsi
            # self.gt = gt

        def __call__(self):
            print('original sizes')
            print(self.hsi.shape)# (83,86,204)
            self.hsi_padded = adjust_image_dimensions(self.hsi)

            hsi_topleft, hsi_topright, hsi_bottomleft, hsi_bottomright = split_image_into_quadrants(self.hsi_padded)

            train_dataset1 = HSIDataset(hsi_topleft)
            train_dataset2 = HSIDataset(hsi_topright)
            train_dataset3 = HSIDataset(hsi_bottomleft)
            train_dataset = torch.utils.data.ConcatDataset([train_dataset1, train_dataset2, train_dataset3])

            test_hsi = hsi_bottomright
            test_dataset = HSIDataset(test_hsi)

            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            # Check batch shapes in DataLoader
            for hsi_batch in train_loader:
                print("HSI Batch Shape:", hsi_batch.shape)  # Should be (batch_size, 1, H, W)
                break  

            for hsi_batch in test_loader:
                print("HSI Batch Shape:", hsi_batch.shape)  # Should be (batch_size, 1, H, W)
                break 
            return train_loader,test_loader
        
        def padded_image(self):
            return self.hsi_padded



class HSIDataset(Dataset):
    def __init__(self, hsi):
        """
        Args:
            hsi (numpy.ndarray): Hyperspectral image data with shape (H, W, C).
            gt (numpy.ndarray): Ground truth data with shape (H, W).
        """
        self.hsi = torch.from_numpy(hsi).float()  # (H, W, C)

    def __len__(self):
        return self.hsi.shape[2]  # Number of spectral bands (C)

    def __getitem__(self, idx):
        band = self.hsi[:, :, idx].unsqueeze(0)  # (H, W) -> (1, H, W)
        return band
    
def split_image_into_quadrants(hsi):
    H, W, _ = hsi.shape
    h_mid = H // 2
    w_mid = W // 2

    hsi_topleft = hsi[:h_mid, :w_mid, :]
    hsi_topright = hsi[:h_mid, w_mid:, :]
    hsi_bottomleft = hsi[h_mid:, :w_mid, :]
    hsi_bottomright = hsi[h_mid:, w_mid:, :]

    return hsi_topleft, hsi_topright, hsi_bottomleft, hsi_bottomright

def adjust_image_dimensions(hsi):
    H, W, C = hsi.shape
    pad_h = (16 - H % 16) if H % 16 != 0 else 0
    pad_w = (16 - W % 16) if W % 16 != 0 else 0
    
    hsi_padded = np.pad(hsi, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    
    return hsi_padded
    
def encode_quadrant(model, quadrant):
    model.eval()  
    with torch.no_grad():  
        quadrant_tensor = torch.from_numpy(quadrant).float().unsqueeze(0).permute(0, 3, 1, 2).to('cuda' if torch.cuda.is_available() else 'cpu')
        print(quadrant.shape)

        encoded_quadrant = model.encoder(quadrant_tensor)
        
    return encoded_quadrant.cpu()  

def get_full_image_encoding(model, hsi_padded_full):
    hsi_topleft, hsi_topright, hsi_bottomleft, hsi_bottomright = split_image_into_quadrants(hsi_padded_full)
    
    encoded_topleft = encode_quadrant(model, hsi_topleft)
    encoded_topright = encode_quadrant(model, hsi_topright)
    encoded_bottomleft = encode_quadrant(model, hsi_bottomleft)
    encoded_bottomright = encode_quadrant(model, hsi_bottomright)
    
    encoded_top = torch.cat((encoded_topleft, encoded_topright), dim=3)
    encoded_bottom = torch.cat((encoded_bottomleft, encoded_bottomright), dim=3)
    full_encoded_image = torch.cat((encoded_top, encoded_bottom), dim=2)
    
    return full_encoded_image
def main():

    model = Autoencoder()

    #abstract out path later
    salinasA_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_corrected.mat'
    salinasA_gt_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_gt.mat'
    #and loadHSI stuff..

    X, M, N, D, HSI, GT, Y, n, K = loadHSI(salinasA_path, salinasA_gt_path, 'salinasA_corrected', 'salinasA_gt')
    #i am redefnig HSI here bc it became weired from some of the other stuff we did in load HSI where we were flattening the image

    #above, loaded data, where the data is HSI and ground truth is GT. then shape of the HSI data is 83 x 86 x 204 and GT is 83 x 86
    
    HSI = X.reshape((M, N, D)) 
    data = DataFormatter(HSI)

    train_loader,test_loader = data()
    hsi_padded = data.padded_image()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the autoencoder
    num_epochs = 30
    print(len(train_loader))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (hsi_batch) in enumerate(train_loader):
            hsi_batch = hsi_batch.to(device)

            optimizer.zero_grad()
            outputs = model(hsi_batch)
            loss = mse(outputs, hsi_batch)  
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:  
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for hsi_batch in test_loader:
                hsi_batch = hsi_batch.to(device)
                outputs = model(hsi_batch)
                loss = mse(outputs, hsi_batch)
                total_loss += loss.item()

        print(f'Validation Loss after Epoch [{epoch + 1}/{num_epochs}]: {total_loss / len(test_loader):.4f}')

#here i should comine the pieces to make a full encoded image..
    # full_image_encoding = get_full_image_encoding(model,hsi_padded,gt_padded)
    # print("Full Image Encoding Shape:", full_image_encoding.shape)
    # save_path = 'encoded_full_image.pt'
    # torch.save(full_image_encoding, save_path)
    # print(f"Encoded representation saved to {save_path}")

    torch.save(model.state_dict(), 'conv_autoencoder.pth')


if __name__ == "__main__":
    main()