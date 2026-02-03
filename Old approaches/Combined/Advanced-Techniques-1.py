import torch
import torch.nn as nn
import time
from torchvision import models

#import torch
#import torch.nn as nn
#import time
#from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup: A standard ResNet50
model = models.resnet50().to(device)
model = torch.compile(model)
#model = models.resnet50().to(device)
#model = torch.compile(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#criterion = nn.CrossEntropyLoss()

scaler = torch.amp.grad_scaler(device)
#scaler = torch.amp.grad_scaler(device)
#scaler = torch.amp.grad_scaler(device)
#scaler = torch.amp.grad_scaler(device)
#scaler = torch.amp.grad_scaler(device)

# Dummy Data (Batch Size 16 is the limit for 16GB VRAM in FP32)
inputs = torch.randn(16, 3, 224, 224).to(device)
labels = torch.randint(0, 1000, (16,)).to(device)

print("Starting Training...")

from torch.profiler import profile, ProfilerActivity
#from torch.profiler import profile, ProfilerActivity
#from torch.profiler import profile, ProfilerActivity
#from torch.profiler import profile, ProfilerActivity
#from torch.profiler import profile, ProfilerActivity

#with profile(activities=[ProfilerActivity.CUDA], schedule=torch.profiler.schedule(wait=1, warmup=1, active=3)) as prof:
#with profile(activities=[ProfilerActivity.CUDA], schedule=torch.profiler.schedule(wait=1, warmup=1, active=3)) as prof:
#with profile(activities=[ProfilerActivity.CUDA], schedule=torch.profiler.schedule(wait=1, warmup=1, active=3)) as prof:
with profile(activities=[ProfilerActivity.CUDA], schedule=torch.profiler.schedule(wait=1, warmup=1, active=3)) as prof:
    for step in range(5):
        
        optimizer.zero_grad(set_to_none=True)
        
        # CRIME #1: The "Guesswork" Profiling
        # The engineer tries to time the forward pass using Python time
        # Why is this misleading for GPU code? What tool should they use instead?
        # t1 = time.time()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward
        scaler.step(optimizer)
        scaler.update()

        prof.step()
        
        # CRIME #2: Missing Compilation (The "Free Lunch")
        # This is standard PyTorch "Eager Mode". 
        # Since PyTorch 2.0, there is a one-line change here that speeds up standard models by ~30%.
        # outputs = model(inputs)
        
        
        
        # CRIME #3: Precision Waste
        # We are running entirely in Float32 (32-bit).
        # Modern GPUs (Tensor Cores) run 2x-3x faster in 16-bit.

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))