from sklearn.ensemble import RandomForestClassifier

def rf_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=100,
                                  random_state=42,
                                  class_weight='balanced',
                                  n_jobs=-1)
