import joblib
import os

def save_model(model, file_name="../models/iris_classifier.pkl"):
    """Save the trained model as a .pkl file."""
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    joblib.dump(model, file_name)
    print(f"💾 Model saved as {file_name}")
