import torch
import torch.nn as nn
import time
import os

from torch.utils.data import DataLoader, TensorDataset

# CRIME #1: Hardcoded Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A pre-trained ResNet-like model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.to(device)
model.eval()

#model = torch.hub.load('', 'resnet18', pretrained=True)
#model.to(device)
#model.eval()

#model = 
#model.to(device)
#model.eval()

# Dummy dataset of 100 images
#dataset = [torch.randn(3, 224, 224) for _ in range(100)] 

tensor_x = torch.stack([torch.randn(3, 224, 224) for _ in range(100)])
my_dataset = TensorDataset(tensor_x)

inference_loader = DataLoader(my_dataset, batch_size=32, num_workers=os.cpu_count())

results = []

print("Starting Inference...")
start = time.time()

# CRIME #2: Sequential Processing (The "Batch Size of 1" Trap)
# Loop runs 100 times. GPU wakes up, processes 1 image, goes to sleep.
with torch.no_grad():
    for batch in inference_loader:
        images = batch[0]

        images = images.to(device, non_blocking=True)

        outputs = model(images)

        results.append(outputs.cpu())

final_results = torch.cat(results)

end = time.time()
print(f"Total time: {end - start:.4f} seconds")