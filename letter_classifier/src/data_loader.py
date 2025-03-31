import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

class TorchDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        X = df.iloc[:, 1:].values.reshape(-1, 28, 28)
        y = df.iloc[:, 0].astype(int).values
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
    

def parse_data(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].astype(int).values  
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
