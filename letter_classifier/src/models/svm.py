from sklearn.svm import SVC

def svm_model() -> SVC:
    return SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
