from numpy import ndarray

from typing import Protocol, Any

class SklearnClassifier(Protocol):
    def fit(self, X: ndarray, y: ndarray) -> Any: ...
    def predict(self, X: ndarray) -> ndarray: ...

def train_sklearn_model(model: SklearnClassifier, X_train: ndarray,
                        y_train: ndarray, X_test: ndarray) -> ndarray:
    model.fit(X_train, y_train)
    return model.predict(X_test)
