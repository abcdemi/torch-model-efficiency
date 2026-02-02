import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import math

class ConfigurableModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Output layer
        )

    def forward(self, x):
        return self.net(x)

def train_ultimate_disaster(features_np, targets_np, epochs=10):
    # features_np shape: (10000, 50)
    # targets_np shape:  (10000,)

    dataset = TensorDataset(torch.from_numpy(features_np).float(), torch.from_numpy(features_np).float().unsqueeze(1))

    loader = DataLoader(
        dataset,
        batch_size=32,
        pin_memory=True,
        num_workers=4,
        shuffle=True
    )
    
    model = ConfigurableModel(features_np.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # MISTAKE 3: The Memory Time Bomb
    all_losses = []

    model.train()
    
    for epoch in range(epochs):
        # MISTAKE 4: The Optimizer Reset
        
        
        # MISTAKE 5: Manual Batching (The CPU Bottleneck)
        for i, (bx, by) in enumerate(loader):
            
            # MISTAKE 6: On-the-fly Transfer (Kernel Launch Overhead)
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            
            # MISTAKE 7: Redundant Move
            optimizer.zero_grad(set_to_none=True)
            
            # Forward
            pred = model(b_x) # Shape: (32, 1)
            
            # MISTAKE 8: The Silent Broadcasting Bug (Math)
            # batch_y shape is (32). prediction is (32, 1).
            # We are manually calculating Error to add custom logic
            #error = prediction - batch_y 
            loss = criterion(pred, b_y)
            
            # MISTAKE 9: Manual Regularization (Efficiency)
            
            
            # MISTAKE 10: Graph Retention
            loss.backward()
            optimizer.step()
            
            # MISTAKE 11: The Sync Barrier
            print(f"Batch {i}, Loss: {loss.item()}")
            
            # (See Mistake 3)
            all_losses.append(loss.item())