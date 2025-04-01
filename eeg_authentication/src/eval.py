from sklearn.metrics import classification_report, accuracy_score
from numpy import ndarray
import torch
from torch import device
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Tuple

def evaluate_torch(model: Module, dataloader: DataLoader,
                   device: device) -> Tuple[list[int], list[int]]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X)
            predictions = torch.argmax(outputs, dim=1).cpu()
            y_true.extend(y.numpy())
            y_pred.extend(predictions.numpy())
    return y_true, y_pred

def evaluate_model(y_true: ndarray, y_pred: ndarray, model_name: str = "") -> None:
    print(f"\n=== {model_name} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["Known", "Unknown"]))
