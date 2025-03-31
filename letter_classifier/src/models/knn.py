from sklearn.neighbors import KNeighborsClassifier

def knn_model() -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=5)
