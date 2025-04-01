import os

from src.data_loader import parse_data
from src.models.random_forest import rf_model
from src.wrappers import train_sklearn_model
from src.eval import evaluate_model

CSV_PATH = "data/autentykacja_eeg.csv"

def run_rf(csv_path: str, blink_flag: bool) -> None:
    X_train, X_test, y_train, y_test = parse_data(csv_path, blink_flag=blink_flag)
    model = rf_model()
    y_pred = train_sklearn_model(model, X_train, y_train, X_test)
    evaluate_model(y_test, y_pred, model_name=f"Random Forest (blink_flag={blink_flag})")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_rf(CSV_PATH, blink_flag=False)
    run_rf(CSV_PATH, blink_flag=True)
