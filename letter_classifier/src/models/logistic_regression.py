from sklearn.linear_model import LogisticRegression

def logistic_model() -> LogisticRegression:
    return LogisticRegression(solver='lbfgs', max_iter=1000)
