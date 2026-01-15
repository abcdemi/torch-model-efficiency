import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimppleImageDataset(Dataset):
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        dummy_image = torch.randn(3, 224, 224)
        label = torch.randint(0, 10, (1,)).item()
        return dummy_image, label
    
dataset = SimppleImageDataset()

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

feature_extractor = models.resnet50(pretrained=True).to(device)
classifier = nn.Linear(1000, 10).to(device)

for param in feature_extractor.parameters():
    param.requires_grad = False
feature_extractor.eval()

optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)
#for param in feature_extractor.parameters():
    #param.requires_grad=False
#for param in feature_extractor.parameters():
    #param.requires_grad=False

# We freeze the ResNet because we only want to train the classifier

print("Starting training...")

for epoch in range(5):
    for inputs, labels in loader:

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            features = feature_extractor(inputs)
        
        outputs = classifier(features)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        loss.backward()
        optimizer.step()