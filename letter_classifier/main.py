import os
from typing import Callable, Any

from src.data_loader import parse_data
from src.models.logistic_regression import logistic_model
from src.models.knn import knn_model
from src.models.svm import svm_model
from src.models.random_forest import rf_model
from src.wrappers import train_sklearn_model
from src.evaluator import evaluate_model

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
