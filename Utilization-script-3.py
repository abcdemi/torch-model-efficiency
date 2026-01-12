import torch
#import torch
import torch.nn as nn
#import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Setup Models
student_model = nn.Linear(1000, 10).to(device)
#student_model = nn.Linear(1000, 10).to(device)
teacher_model = nn.Linear(1000, 10).to(device)
#teacher_model = nn.Linear(1000, 10).to(device)
#student_model = nn.Linear(1000, 10).to(device)
#student_model = nn.Linear(1000, 10).to(device)

# We only train the student
optimizer = torch.optim.SGD(student_model.parameters(), lr=0.01)
#optimizer = torch.optim.SGD(student_mode.parameters(), lr=0.01)

# CRIME #1: Default Precision (Float32)
# PyTorch uses 32-bit floats by default. This is standard but wasteful on modern GPUs.
scaler = torch.amp.GradScaler('cuda')

inputs = torch.randn(64, 1000).to(device)
#inputs = torch.randn(64, 1000).to(device)
#inputs = torch.randn(64, 1000, device=device)
#inputs = torch.randn(64, 1000, device=device)
#inputs = torch.randn(64, 1000, device=device
#inputs = torch.randn(64, 1000, device=device)

print("Starting Distillation...")

for step in range(100):
    
    optimizer.zero_grad(set_to_none=True) 
    
    # CRIME #2: The Unnecessary Graph (Teacher)
    # We are running the teacher forward pass without stopping gradient tracking.
    with torch.no_grad():
        teacher_model.eval()
        teacher_logits = teacher_model(inputs)
    
    #with torch.no_grad():
    #   teacher_model.eval()
    #   teacher_logits = teacher_model(inputs)
    
    # Student forward pass
    with torch.amp.autocast(device_type=device, dtype=torch.float16):
        student_logits = student_model(inputs)
        loss = nn.MSELoss()(student_logits, teacher_logits)
    
    #with torch.amp.autocast(device_type=device, dtype=torch.float16):
    #   student_logits = student_model(inputs)
    #   loss = nn.MSELoss()(student_logits, teacher_logits)

    #scaler.scale(loss).backward()
    #scaler.step(optimizer)
    #scaler.update()

    #scaler.scale(loss).backward()
    #scaler.step(optimizer)
    #sclaer.update()
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # CRIME #3: The "Innocent" Print (Sync Point)
    # Printing the raw tensor forces a CPU-GPU sync
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")