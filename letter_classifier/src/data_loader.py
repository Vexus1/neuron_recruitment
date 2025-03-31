import pandas as pd
from sklearn.model_selection import train_test_split

def parse_data(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].astype(int).values  
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
