import torch
import torch.nn as nn
import numpy as np
#import torch
#import torch.nn as nn
#import numpy as np

# CRIME #1: Hardware Assumption
# The engineer hardcoded the GPU ID. What if we are on a multi-GPU node or a CPU-only debugging machine?
#device = torch.device("cuda:0") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = nn.Linear(1000, 10).to(device)
model = nn.Linear(1000, 10).to(device)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()

# A list to store loss history for plotting later
loss_history = []

print("Starting training...")

for epoch in range(100):
    # Imagine a standard DataLoader here
    for i, (inputs, targets) in enumerate(dataloader):
        
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Standard zero_grad
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            accuracy = (outputs.argmax(dim=1) == targets).float().mean()


        # Forward pass
        #outputs = model(inputs)
        #loss = criterion(outputs, targets)
        
        #loss.backward()
        #optimizer.step()
        
        # CRIME #2: The Performance Killer (Synchronization)
        # The engineer wants to calculate accuracy for logging
        # They move data to CPU to use NumPy
        #predictions = outputs.argmax(dim=1).cpu().numpy()
        #accuracy = np.mean(predictions == targets.cpu().numpy())
        
        # CRIME #3: The Memory Leak (The Time Bomb)
        # Storing the loss to plot it later
        loss_history.append(loss.item())
        
        # Standard zero_grad
        #optimizer.zero_grad() 

    print(f"Epoch {epoch} finished.")