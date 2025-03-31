import torch
import torch.nn as nn
import torch.optim as optim
from torch import device as TorchDevice
from torch.utils.data import DataLoader
from torch.nn import Module
from typing import Tuple
import pandas as pd

def train_torch(model: Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: TorchDevice,
                epochs: int = 5,
                log_path = "outputs/cnn_training_log.csv") -> Module:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_loss, val_acc = evaluate_on_validation(model, val_loader, device, criterion)
        train_losses.append(total_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss:.4f},"
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    log_df = pd.DataFrame({
        "epoch": list(range(1, epochs + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_accuracy": val_accuracies
    })
    log_df.to_csv(log_path, index=False)
    return model

def evaluate_on_validation(model: Module,
                           val_loader: DataLoader,
                           device: TorchDevice,
                           criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            predictions = torch.argmax(output, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy
