import torch
import torch.nn as nn

def train_and_validate_fixed(model, train_loader, val_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CosineEmbeddingLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    accumulation_steps = 4

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for i, (x1, x2, label) in enumerate(train_loader):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output1, output2 = model(x1, x2)
            loss = criterion(output1, output2, label) / accumulation_steps
            loss.backward()

            if (i+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * accumulation_steps

        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train =  total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for x1, x2, label in val_loader:
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                output1, output2 = model(x1, x2)
                loss = criterion(output1, output2, label)

                total_val_loss += loss.item()

        print(f"Epoch {epoch}: Train {avg_train:.4f}, Val {total_val_loss/len(val_loader):.4f}")