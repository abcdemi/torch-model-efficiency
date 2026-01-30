import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, Dataset
import os

# CRIME #1: Hardware Setup
# The engineer assumes we always have an NVIDIA GPU.
#device = "cuda" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class AugmentedDataset(Dataset):
    def __len__(self): return 1000
    def __getitem__(self, index):
        time.sleep(0.005)
        return torch.randn(1024), torch.randn(10)

dataset = AugmentedDataset()

loader = DataLoader(dataset, batch_size=64, num_workers=os.cpu_count())

model = nn.Linear(1024, 10).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Simple Model
#model = nn.Linear(1024, 10).cuda()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Dummy Data Generator
#def get_batch():
#    return torch.randn(64, 1024), torch.randn(64, 10)

print("Starting Training...")

for epoch in range(5):
    # --- TRAINING PHASE ---
    model.train()
    
    # CRIME #2: The CPU Bottleneck
    # We get a batch, but then we do some "heavy augmentation" on CPU.
    # In real life, this is complex numpy math or image rotation.
    #inputs, targets = get_batch()
    
    # Simulating a heavy CPU task (e.g., 50ms of processing)
    #time.sleep(0.05) 
    for inputs, targets in loader:
        # Moving to GPU
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # --- VALIDATION PHASE ---
    # We want to check accuracy, but we don't need to update weights.
    
    # CRIME #3: The Memory Leak
    model.eval() # We set the model to eval mode... is that enough?
    
    with torch.no_grad():
        val_inputs = torch.randn(64, 1024).to(device)
        val_targets = torch.randn(64, 10).to(device)

        val_outputs = model(val_inputs)
        val_loss = nn.MSELoss()(val_outputs, val_targets)
    
    print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")