import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import random
import os

# --- PROFILER IMPORTS ---
from torch.profiler import profile, record_function, ProfilerActivity, schedule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = {f"word_{i}": i for i in range(10000)}

# --- DATASET & COLLATE ---
class TextDataset(Dataset):
    def __init__(self):
        self.data = [f"word_{random.randint(0,9999)} " * 50 for _ in range(5000)]
        self.labels = [random.randint(0, 1) for _ in range(5000)]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def optimized_collate(batch):
    text_list, label_list = zip(*batch)
    tokenized_list = [torch.tensor([vocab.get(w, 0) for w in text.split()], dtype=torch.long) for text in text_list]
    inputs = pad_sequence(tokenized_list, batch_first=True, padding_value=0)
    labels = torch.tensor(label_list, dtype=torch.long)
    return inputs, labels

# --- MODEL ---
class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 512)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True), num_layers=6)
        self.fc = nn.Linear(512, 2)
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.fc(x.mean(dim=1))

if __name__ == '__main__':
    # 1. SETUP
    model = TransformerClassifier().to(device)
    #model = torch.compile(model) # <--- Optimization Restored
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    train_loader = DataLoader(
        TextDataset(), batch_size=64, shuffle=True,
        collate_fn=optimized_collate, num_workers=4,
        pin_memory=True, drop_last=True
    )

    print(f"Starting Profiled Training on {device}...")

    # 2. PROFILER CONTEXT MANAGER
    # We trace both CPU and CUDA to find bottlenecks
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], # <--- Crucial for GPU timing
        schedule=schedule(wait=5, warmup=5, active=10, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler_demo'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        
        for i, (inputs, labels) in enumerate(train_loader):
            
            # Optional: Add custom labels to the timeline
            with record_function("data_transfer"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with record_function("forward_pass"):
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            
            with record_function("backward_pass"):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Mark step end
            prof.step()
            
            # Stop early for demo
            if i >= 25: 
                break

    # 3. PRINT RESULTS
    print("\nPROFILER SUMMARY:")
    # We sort by CUDA time to see the most expensive GPU kernels
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))