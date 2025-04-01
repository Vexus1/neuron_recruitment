import pandas as pd
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class MLPDataset(Dataset):
    def __init__(self, csv_path: str, blink_flag: bool = False):
        X_train, _, y_train, _ = parse_data(csv_path, blink_flag=blink_flag)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.y = torch.tensor(y_train.astype(int), dtype=torch.long)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def parse_data(csv_path: str, 
               blink_flag: bool = False) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    df = pd.read_csv(csv_path, sep=';')
    # print(df[df["BlinkStrength"] == -1]["Flag"].value_counts(normalize=True)) 
    # 1 ~ 0.52, 0 ~ 0.47
    if blink_flag:
        df["BlinkDetected"] = (df["BlinkStrength"] != -1).astype(int)
    if (df["BlinkStrength"] == -1).any():
        blink_median = df.loc[df["BlinkStrength"] != -1, "BlinkStrength"].median()
        df["BlinkStrength"] = df["BlinkStrength"].replace(-1, blink_median)
    # print(df.head(n = 25))
    X = df.drop(columns=["Flag"]).values
    y = df["Flag"].values
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

