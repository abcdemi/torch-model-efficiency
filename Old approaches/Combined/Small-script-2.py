import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        return self.fc(x)

def train_inefficient(features_np, targets_np, epochs=5):
    dataset = TensorDataset(torch.from_numpy(features_np), torch.from_numpy(targets_np))

    batch_size = 64
    target_batch_size = 256
    accumulation_steps = target_batch_size // batch_size

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Assume 'dataset' is a standard Python list of numpy arrays
    model = LinearModel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)

    #optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)
    criterion = nn.MSELoss()
    #criterion = nn.MSELoss()
    #model.train()

    #criterion = nn.MSELoss()
    #model.train()
    
    # 1. The Time Bomb
    all_losses = []

    model.train()

    for epoch in range(epochs):
        # 2. The CPU Bottleneck
        for i in (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            prediction = model(batch_x)

            loss = criterion(prediction, batch_y) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            
            all_losses.append(loss.item() *  accumulation_steps)
            
            # 6. The Sync Barrier
            if i % 100 == 0:
                print(f"Iter {i}, Loss: {loss.item()}")
            
        if (i + 1) % accumulation_steps != 0:
             optimizer.step()
             optimizer.zero_grad(set_to_none=True)