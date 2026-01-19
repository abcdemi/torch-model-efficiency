import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import time
import random
from PIL import Image

# CRIME 1: Hardware Agnosticism
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy Data: 1000 samples of (Image, Text)
class MultiModalDataset(Dataset):
    def __init__(self):
        self.data = list(range(1000))
    def __len__(self): return 1000
    
    def __getitem__(self, idx):
        # Simulate loading a large image
        # CRIME 2: Heavy CPU Transform on every access
        # This resize happens in the main process if num_workers=0
        img = torch.randn(3, 1024, 1024) 
        resize = T.Resize((224, 224))
        img = resize(img)
        
        # Simulate text
        text = torch.randint(0, 10000, (20,)) # 20 tokens
        
        return img, text, random.randint(0, 1)

# A simple Two-Tower Model (Image Encoder + Text Encoder)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_enc = nn.Sequential(
            nn.Conv2d(3, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 128, 3), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*220*220, 256)
        )
        self.text_enc = nn.Embedding(10000, 256)
        self.head = nn.Linear(512, 2)
        
    def forward(self, img, text):
        i_emb = self.img_enc(img)
        t_emb = self.text_enc(text).mean(dim=1)
        combined = torch.cat([i_emb, t_emb], dim=1)
        return self.head(combined)

# CRIME 3: Missing Compilation (PyTorch 2.0)
model = MyModel().to(device)
model = torch.compile(model)

# CRIME 4: Data Loader Bottleneck
# Defaults: num_workers=0 (Main Process), pin_memory=False
loader = DataLoader(
    MultiModalDataset(), 
    batch_size=32, 
    shuffle=True,
    pin_memory=True,
    num_workers=4
)

scaler = torch.amp.GradScaler("cuda")

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("Starting training...")

model.train()

for epoch in range(3):
    # CRIME 5: Bad Profiling
    torch.cuda.synchronize()
    t0 = time.time()
    
    for i, batch in enumerate(loader):
        # CRIME 6: Manual Unpacking & Movement
        # "batch" is a list of [imgs, texts, labels] from the loader
        # We are moving them to GPU one by one? 
        # Actually, standard loaders return stacked tensors, but without pin_memory, 
        # this .to() call is synchronous and slow.
        imgs = batch[0].to(device, non_blocking=True)
        texts = batch[1].to(device, non_blocking=True)
        labels = batch[2].to(device, non_blocking=True)
        
        # CRIME 7: Gradient Reset Efficiency
        optimizer.zero_grad(set_to_none=True)
        
        # CRIME 8: Precision Waste (Float32)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(imgs, texts)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # CRIME 9: Bad Profiling (Stopping the timer)
    # GPU is async! This timer will report finish before GPU is done.
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Epoch {epoch} done in {t1-t0:.4f}s")