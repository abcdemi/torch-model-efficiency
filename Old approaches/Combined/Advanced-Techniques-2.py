import torch
import torch.nn as nn
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A dummy "Heavy" Model (simulating a U-Net)
# In reality, this would be segmentation_models_pytorch.Unet(...)
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(256, 10, kernel_size=1) # Output: 10 classes per pixel
).to(device)

# FIX #2: COMPILE
# "I'm adding torch.compile to fuse the Conv+ReLU layers."
model = torch.compile(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

scaler = torch.amp.GradScaler('cuda')

# Big Inputs (High Resolution Medical Images)
inputs = torch.randn(8, 3, 512, 512).to(device)
targets = torch.randint(0, 10, (8, 512, 512)).to(device)

print("Starting Training...")

from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA], schedule=torch.profiler.schedule(wait=1, warmup=1, active=3)) as prof:

    for step in range(20):
        
        optimizer.zero_grad(set_to_none=True)
        
        # CRIME #1: The Lying Timer
        # The colleague swears the forward pass is instantaneous (0.0001s).
        # Why is this print statement misleading?
        #t1 = time.time()
        
        # CRIME #2: Missing Compilation
        # This runs in Eager Mode. A Segmentation model is perfect for fusion optimization.
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        prof.step()
        
        #t2 = time.time()
        #print(f"Forward pass: {t2 - t1:.6f}s") 
        
        # CRIME #3: The Debugging Anchor
        # The colleague left a debug print to check output shapes.
        # Why is this specific line destroying the training throughput?
        #print(f"Output shape: {outputs.cpu().shape}")
        
        # CRIME #4: Precision Bloat
        # Segmentation masks are huge. Storing them in Float32 is killing memory.
        #loss.backward()
        #optimizer.step()
        
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))