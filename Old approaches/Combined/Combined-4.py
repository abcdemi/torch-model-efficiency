import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
import os

class FixedModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 64)
        self.layer2 = nn.Linear(64, 1)
        # CRIME: Brain (Bad Init)
        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity='linear')

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        # CRIME: Brain (Output Activation Trap)
        return x

def train_nightmare(data_np, targets_np, epochs=5):
    # data: numpy array (10000, 50)
    # targets: numpy array (10000,)

    tensor_x = torch.from_numpy(data_np).float()
    tensor_y = torch.from_numpy(targets_np).float().unsqueeze(1)

    dataset = TensorDataset(tensor_x, tensor_y)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FixedModel(data_np.shape[1]).to(device)
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    scaler = torch.amp.GradScaler("cuda")
    
    # CRIME: Leaks (History Bomb)
    loss_history = [] 

    model.train()
    
    for epoch in range(epochs):
        # CRIME: Brain (Optimizer Reset)
        
        # CRIME: Kitchen (Manual Batcher & Solo Worker)
        #batch_size = 32
        for i, (bx, by) in enumerate(loader):
            
            # CRIME: Kitchen (Loop Tokenizer/Processing)
            # Simulating heavy processing in main thread            
            # CRIME: Traffic (The Stop-and-Go)
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            
            # CRIME: Traffic (The Hardcoded Device & Redundant Move)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                preds = model(bx)
                loss = criterion(preds, by)

            # CRIME: Engine (The Slow Eraser)
            optimizer.zero_grad(set_to_none=True) 
            
            
            
            # CRIME: Brain (The Silent Broadcasting Bug)
            # preds is (32, 1), batch_y is (32) -> Matrix broadcasting
        
            
            # CRIME: Leaks (Graph Retention)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # CRIME: Traffic (The Sync Point)
            if i % 100 == 0:
                print(f"Loss: {loss.item()}") 
            
            # CRIME: Leaks (History Bomb)
            loss_history.append(loss.item())