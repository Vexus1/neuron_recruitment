import os

import torch
from torch.utils.data import DataLoader, random_split

from src.data_loader import parse_data, MLPDataset
from src.models.random_forest import rf_model
from src.wrappers import train_sklearn_model
from src.eval import evaluate_model
from src.models.mlp import MLPClassifier
from src.torch_trainer import train_torch
from src.eval import evaluate_torch

CSV_PATH = "data/autentykacja_eeg.csv"

def run_rf(csv_path: str, blink_flag: bool) -> None:
    X_train, X_test, y_train, y_test = parse_data(csv_path, blink_flag=blink_flag)
    model = rf_model()
    y_pred = train_sklearn_model(model, X_train, y_train, X_test)
    evaluate_model(y_test, y_pred, model_name=f"Random Forest (blink_flag={blink_flag})")

def run_mlp(csv_path: str, blink_flag: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MLPDataset(csv_path, blink_flag=blink_flag)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    model = MLPClassifier(input_size=dataset.X.shape[1])
    train_torch(model, train_loader, val_loader, device, epochs=50, patience=5)
    y_true, y_pred = evaluate_torch(model, val_loader, device)
    evaluate_model(y_true, y_pred, model_name=f"MLP (blink_flag={blink_flag})")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_rf(CSV_PATH, blink_flag=False)
    run_rf(CSV_PATH, blink_flag=True)
    run_mlp(CSV_PATH, blink_flag=False)
    # run_mlp(CSV_PATH, blink_flag=True)
