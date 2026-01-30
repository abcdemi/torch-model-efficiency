import torch
from torch.utils.data import DataLoader, TensorDataset

def train_fixed(features, targets, epochs=5):
    # 0. SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # FIX: Kitchen (Efficient Pipeline)
    # Prepare data properly to handle (N, 1) targets
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(targets).float().unsqueeze(1) # FIX: Broadcasting Logic
    )
    
    # FIX: Kitchen (Workers & Pinning)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, 
                        num_workers=4, pin_memory=True)
    
    # FIX: Precision (Keep Model in FP32)
    # We DO NOT call .half(). We keep Master Weights in FP32.
    model = HousePriceModel().to(device)
    
    # FIX: Brain (Optim Init Once)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # FIX: Precision (The Scaler)
    # Handles the "Dynamic Range" issue
    scaler = torch.cuda.amp.GradScaler()
    
    loss_history = []
    
    # TRAINING LOOP
    for epoch in range(epochs):
        model.train() # FIX: Brain (Enable BN updates)
        
        for bx, by in loader:
            # FIX: Traffic (Async Transfer)
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # FIX: Precision (Autocast Context)
            # "Safe" ops (BN, Loss) run in FP32. "Fast" ops (MatMul) run in FP16.
            with torch.amp.autocast("cuda"):
                preds = model(bx)
                loss = criterion(preds, by)
            
            # FIX: Precision (Scaled Backward)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # FIX: Leaks (Detach from Graph)
            loss_history.append(loss.item())

    # VALIDATION PHASE
    model.eval() # FIX: Brain (Freeze BN stats)
    #model.eval()
    val_loss = 0.0
    
    # FIX: Leaks (Disable Graph Construction)
    #with torch.no_grad():
    with torch.no_grad():
        for i in range(100):
            val_x = torch.randn(64, 50).to(device)
            val_y = torch.randn(64, 1).to(device)
            
            # Autocast works in inference too!
            with torch.amp.autocast("cuda"):
                out = model(val_x)
                l = criterion(out, val_y)
            
            # with torch.amp.autocast("cuda"):
            
            # FIX: Leaks (Accumulate Float)
            val_loss += l.item() 
            
    print(f"Avg Val Loss: {val_loss / 100}")