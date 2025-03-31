from numpy import ndarray
from sklearn.preprocessing import StandardScaler

from typing import Protocol, Any

class SklearnClassifier(Protocol):
    def fit(self, X: ndarray, y: ndarray) -> Any: ...
    def predict(self, X: ndarray) -> ndarray: ...


def scale_data(X_train: ndarray, X_test: ndarray) -> tuple[ndarray, ndarray]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_sklearn_model(model: SklearnClassifier, X_train: ndarray,
                        y_train: ndarray, X_test: ndarray,
                        use_scaling: bool = False) -> ndarray:
    if use_scaling:
        X_train, X_test = scale_data(X_train, X_test)
    model.fit(X_train, y_train)
    return model.predict(X_test)
