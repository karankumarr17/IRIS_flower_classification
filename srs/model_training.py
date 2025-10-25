from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    """Train a RandomForestClassifier on the Iris dataset."""
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    print("✅ Model Trained Successfully!")
    return model
