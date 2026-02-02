import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

class EfficientRNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=20):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_step = lstm_out[:,-1,:]
        return self.fc(last_step)

def train_sequence_model(epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientRNN().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    
    for epoch in range(epochs):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        preds = model(inputs)
        
        loss = criterion(preds, targets)
        
        # Mistake 6: Memory Zombie
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch} complete")