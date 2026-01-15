import torch
import torch.nn as nn
import time
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader

# CRIME #1: Hardware Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup BERT (Heavy Model)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased').to(device)
classifier = nn.Linear(768, 2).to(device) # Binary classification head

#bert = BertModel.from_pretrained('').to(device)
#classifier = nn.Linear(768, 2).to(device)

optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

#optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001)

# 10,000 Dummy sentences
raw_sentences = ["This is a positive sentence " * 10 for _ in range(10000)]

encoded_inputs = tokenizer(raw_sentences, return_tensors="pt", padding=True, truncation=True)
labels = torch.randint(0, 2, (10000,)).long()

dataset = TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'], labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

print("Starting Fine-Tuning...")

for param in bert.parameters():
    param.requires_grad=False

bert.eval()

for batch in loader:

    input_ids, mask, batch_labels = [t.to(device) for t in batch]

    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        outputs = bert(input_ids, attention_mask=mask)
        cls_token = outputs.last_hidden_state[:,0,:]
    
    prediction = classifier(cls_token)
    loss = criterion(prediction, batch_labels)

    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"Step {i}")