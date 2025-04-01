from numpy import ndarray
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score

def evaluate_model(y_true: ndarray, y_pred: ndarray, model_name: str = "") -> None:
    print(f"\n=== {model_name} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["Known", "Unknown"]))
