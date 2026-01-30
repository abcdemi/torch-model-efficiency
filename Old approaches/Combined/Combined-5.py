import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = slow_tokenizer(self.texts[idx])
        return torch.tensor(tokens)

# A simple mock tokenizer
def slow_tokenizer(text):
    # CRIME: Kitchen (The Loop Tokenizer)
    # Simulates heavy CPU work inside the main thread
    time.sleep(0.001) 
    return [ord(c) % 1000 for c in text.split()]

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (Batch, Seq_Len)
        embeds = self.embedding(x)
        
        # lstm_out: (Batch, Seq, Hidden)
        lstm_out, _ = self.lstm(embeds)
        
        # CRIME: Brain (The Padding Polluter)
        # We take the mean of ALL tokens, including Padding (0s).
        # This dilutes the signal with noise.
        avg_pool = torch.mean(lstm_out, dim=1) 
        
        logits = self.fc(avg_pool)
        
        # CRIME: Brain (Double Softmax)
        return self.softmax(logits)

def train_nlp_disaster(raw_texts, labels, epochs=3):
    # raw_texts: List[str] (Variable lengths!)
    # labels: List[int]
    
    vocab_size = 1000
    model = SentimentRNN(vocab_size, 64, 128)
    criterion = nn.CrossEntropyLoss()
    
    # CRIME: Leaks (The History Bomb)
    debug_history = []
    
    for epoch in range(epochs):
        # CRIME: Brain (Optimizer Reset)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # CRIME: Kitchen (Manual Batching)
        batch_size = 32
        for i in range(0, len(raw_texts), batch_size):
            batch_texts = raw_texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # CRIME: Kitchen (The Loop Tokenizer)
            # Tokenizing & Padding inside the training loop (Blocking GPU)
            tokenized = [slow_tokenizer(t) for t in batch_texts]
            max_len = max(len(t) for t in tokenized)
            padded = [t + [0]*(max_len - len(t)) for t in tokenized]
            
            # CRIME: Traffic (Stop-and-Go)
            tensor_x = torch.tensor(padded).cuda()
            tensor_y = torch.tensor(batch_labels).cuda()
            
            # CRIME: Traffic (Redundant Move)
            model.cuda()
            
            # CRIME: Engine (Slow Eraser)
            optimizer.zero_grad()
            
            # CRIME: Engine (Precision Hog - No AMP)
            preds = model(tensor_x)
            
            loss = criterion(preds, tensor_y)
            
            # CRIME: Leaks (Graph Retention)
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # CRIME: Traffic (Sync Barrier)
            print(f"Batch {i}, Loss: {loss.item()}")
            
            # CRIME: Leaks (History Bomb)
            debug_history.append(loss)