import torch
import torch.nn as nn
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A dummy "Super Resolution" Model
# Lots of heavy convolutions and activations
model = nn.Sequential(
    nn.ConvTranspose2d(3, 64, kernel_size=4, stride=2, padding=1), # Upsample 2x
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(256, 3, kernel_size=3, padding=1)
).to(device)
#model = nn.Sequential().to(device)

model = torch.compile(model)
#model = torch.compile(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

scaler = torch.amp.GradScaler('cuda')
#scaler = torch.amp.GradScaler('cuda')

# Inputs: Batch of 32 Low-Res Images (128x128)
inputs = torch.randn(32, 3, 128, 128).to(device)
targets = torch.randn(32, 3, 256, 256).to(device) # High-Res targets

print("Starting training...")

for step in range(10):
    
    optimizer.zero_grad(set_to_none=True)
    #optimizer.zero_grad(set_to_none=True)
    
    # CRIME #1: The "Jittery" Timer
    # The engineer tries to measure just the forward pass.
    # The output varies wildly: 0.00001s, then 0.05s, then 0.00002s.

    torch.cuda.synchronize()
    start = time.time()
    #torch.cuda.synchronize()
    #start = time.time()
    
    # CRIME #2: Eager Execution
    # This is a static model structure. 
    # We are missing the "compile" step that fuses the Conv+ReLU layers.
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    #with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    #outputs = model(inputs)
    #loss = criterion(outputs, targets)
    
    torch.cuda.synchronize()
    end = time.time()
    if step > 0: # Skip first warmup step
        print(f"Time: {end - start:.6f}s")
    
    # CRIME #3: Full Precision Waste
    # We are calculating gradients on massive 256x256 feature maps in Float32.
    # This consumes double the memory needed.
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    #scaler.scale(loss).backward()
    #scaler.step(optimizer)
    #scaler.update()