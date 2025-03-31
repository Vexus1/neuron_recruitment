import os
from typing import Callable, Any

import torch
from torch.utils.data import DataLoader, random_split

from src.data_loader import parse_data
from src.models.logistic_regression import logistic_model
from src.models.knn import knn_model
from src.models.svm import svm_model
from src.models.random_forest import rf_model
from src.wrappers import train_sklearn_model
from src.evaluator import evaluate_model, evaluate_torch
from src.models.cnn import CNNClassifier
from src.data_loader import TorchDataset
from src.torch_trainer import train_torch

def run_cnn() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TorchDataset("data/xyz_dataset.csv")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    model = CNNClassifier()
    train_torch(model, train_loader, val_loader, device, epochs=5)  
    y_true, y_pred = evaluate_torch(model, val_loader, device)  
    evaluate_model(y_true, y_pred, model_name="CNN (PyTorch)")

models: dict[str, tuple[Callable[[], Any], bool]] = {
    "Logistic Regression": (logistic_model, True),
    "KNN": (knn_model, True),
    "Random Forest": (rf_model, False),
    "SVM": (svm_model, True),
}

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    X_train, X_test, y_train, y_test = parse_data("data/xyz_dataset.csv")
    for name, (build, scale) in models.items():
        model = build()
        y_pred = train_sklearn_model(model, X_train, y_train, X_test, use_scaling=scale)
        evaluate_model(y_test, y_pred, model_name=name)
    run_cnn()
