import torch
import torch.nn as nn
from torch.optim import Adam
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from utils import loadHSI, calculate_aligned_accuracy
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class SimpleAutoencoder(nn.Module):
    def __init__(self, hidden_dim=32, init_mode='kaiming'):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(204, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.Conv2d(64, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Flatten(),
            nn.Linear(hidden_dim, 7),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.Sigmoid(),
            nn.Unflatten(1, (hidden_dim, 1, 1)),
            nn.ConvTranspose2d(hidden_dim, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 204, kernel_size=1),
            nn.Sigmoid()
        )
        self._initialize_weights(init_mode)

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

def cluster_loss(encoded, kmeans, GT):
    kmeans_labels = kmeans.predict(encoded)
    kmeans_accuracy = calculate_aligned_accuracy(GT.flatten(), kmeans_labels)
    return -kmeans_accuracy  # We want to maximize accuracy, hence minimizing -accuracy

def train_model(params):
    salinasA_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_corrected.mat'
    salinasA_gt_path = '/Users/seoli/Desktop/DIAMONDS/Tufts2024/data/SalinasA_gt.mat'
    X, M, N, D, HSI, GT, Y, n, K = loadHSI(salinasA_path, salinasA_gt_path, 'salinasA_corrected', 'salinasA_gt')
    
    GT = GT - 1  # Convert to 0-based indexing

    HSI = X.reshape((M, N, D))
    scaler = MinMaxScaler()
    HSI = scaler.fit_transform(HSI.reshape(-1, D)).reshape(M, N, D)
    pixels = HSI.reshape(-1, D, 1, 1)
    pixels = torch.from_numpy(pixels).float()

    dataset = TensorDataset(pixels)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    model = SimpleAutoencoder(hidden_dim=params['hidden_dim'], init_mode=params['init_mode'])
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=1e-5)

    num_epochs = 28
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
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

    model.eval()
    total_loss = 0.0
    encoded_features = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0]
            reconstructed, encoded = model(inputs)
            loss = criterion(reconstructed, inputs)
            total_loss += loss.item()
            encoded_features.append(encoded.cpu().numpy())

    encoded_features = np.concatenate(encoded_features, axis=0).reshape(-1, 7)
    kmeans = KMeans(n_clusters=7, random_state=0).fit(encoded_features)
    clustering_loss = cluster_loss(encoded_features, kmeans, GT)

    return {'loss': clustering_loss, 'status': STATUS_OK}

# Define the search space
space = {
    'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-2)),
    'hidden_dim': scope.int(hp.quniform('hidden_dim', 16, 64, 16)),
    'init_mode': hp.choice('init_mode', ['kaiming', 'xavier'])
}

# Run the optimization
trials = Trials()
best = fmin(fn=train_model,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)

print(f"Best parameters: {best}")
