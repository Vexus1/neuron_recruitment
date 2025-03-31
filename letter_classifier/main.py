import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import parse_data
from src.models.logistic_regression import logistic_model
from src.wrappers import train_sklearn_model
from src.evaluator import evaluate_model
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = parse_data("data/xyz_dataset.csv")
    logistic = logistic_model()
    y_pred = train_sklearn_model(logistic, X_train, y_train,
                                 X_test, use_scaling=True)
    evaluate_model(y_test, y_pred, model_name='logistic')
