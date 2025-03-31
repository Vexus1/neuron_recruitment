import torch 
import torch.nn as nn
import torch.optim as optim
from torch import device
from torch.utils.data import DataLoader
from torch.nn import Module

def train_torch(model: nn.Module, train_loader: DataLoader,
                device: device, epochs: int = 5) -> Module:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
    return model
