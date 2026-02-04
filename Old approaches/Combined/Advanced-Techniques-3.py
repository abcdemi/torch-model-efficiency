import torch
import torch.nn as nn
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A dummy Vision Transformer (Heavy computation)
# In standard PyTorch, this is just a sequence of layers
model = nn.Sequential(
    nn.Conv2d(3, 768, kernel_size=16, stride=16), # Patch Embed
    nn.Flatten(2),
    nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=768, nhead=12), 
        num_layers=12
    ),
    nn.Linear(768, 1000)
).to(device)

model = torch.compile(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Inputs: Batch of 32 images (High resolution)
scaler = torch.amp.GradScaler('cuda')

inputs = torch.randn(32, 3, 224, 224).to(device)
labels = torch.randint(0, 1000, (32,)).to(device)

print("Starting training...")

for step in range(10):
    
    optimizer.zero_grad(set_to_none=True)
    
    # CRIME #1: The "Lying" Timer
    # The engineer wants to see how fast the forward pass is.
    # They wrapped the model call in simple Python timestamps.
    torch.cuda.synchronize()
    t0 = time.time()
    
    # CRIME #2: The Missed "Free Lunch"
    # This model is a standard Transformer. 
    # PyTorch 2.0 has a specific one-line feature that fuses Multi-Head Attention kernels.
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs.mean(dim=2), labels)
    
    torch.cuda.synchronize()
    t1 = time.time()
    if step % 5 == 0:
        print(f"Forward pass time: {t1 - t0:.6f}s") # Prints ~0.0001s (WRONG!)
    
    # CRIME #3: The Precision Hog
    # Transformers love 16-bit precision. 
    # Running them in 32-bit (default) is a waste of A100/H100 Tensor Cores.
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()