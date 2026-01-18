import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import time
import random
import os

# --- 1. ROBUST DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SETUP DUMMY DATA ---
vocab = {f"word_{i}": i for i in range(10000)}

class TextDataset(Dataset):
    def __init__(self):
        self.data = [f"word_{random.randint(0,9999)} " * 50 for _ in range(5000)]
        self.labels = [random.randint(0, 1) for _ in range(5000)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- 2. OPTIMIZED COLLATE_FN ---
def optimized_collate(batch):
    text_list, label_list = zip(*batch)
    tokenized_list = []
    for text in text_list:
        tokens = [vocab.get(w, 0) for w in text.split()]
        tokenized_list.append(torch.tensor(tokens, dtype=torch.long))
    
    inputs = pad_sequence(tokenized_list, batch_first=True, padding_value=0)
    labels = torch.tensor(label_list, dtype=torch.long)
    return inputs, labels

# --- MODEL DEFINITION ---
class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 512)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=6
        )
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.fc(x.mean(dim=1))

# --- MAIN EXECUTION BLOCK ---
# This guard is MANDATORY on Windows when using num_workers > 0
if __name__ == '__main__':
    model = TransformerClassifier().to(device)
    
    # Try an alternative backend that works better on Windows
    try:
        model = torch.compile(model, backend="cudagraphs")
    except:
        print("Compilation failed, falling back to eager mode.")

    # --- 3. COMPILATION ---
    # Note: torch.compile requires a compatible C++ compiler on Windows
    # If this fails, you can comment this line out.

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # --- 4. OPTIMIZED DATALOADER ---
    train_loader = DataLoader(
        TextDataset(), 
        batch_size=64, 
        shuffle=True,
        collate_fn=optimized_collate,
        num_workers=os.cpu_count(),   
        pin_memory=True,
        drop_last=True               
    )

    # --- 5. AMP SCALER ---
    scaler = torch.amp.GradScaler('cuda')

    print(f"Starting Training on {device}...")

    for epoch in range(3):
        torch.cuda.synchronize() 
        t0 = time.time()
        
        for inputs, labels in train_loader:
            # --- 7. ASYNC TRANSFER ---
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # --- 8. FAST ZERO GRAD ---
            optimizer.zero_grad(set_to_none=True)
            
            # --- 9. MIXED PRECISION ---
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # --- 10. SCALED BACKWARD ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        torch.cuda.synchronize() 
        t1 = time.time()
        print(f"Epoch {epoch} Time: {t1 - t0:.4f}s")