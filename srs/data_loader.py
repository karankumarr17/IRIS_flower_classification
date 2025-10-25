import pandas as pd
from sklearn.datasets import load_iris

def load_data():
    """Load the Iris dataset and return as a DataFrame."""
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['species'] = iris.target
    target_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
    data['species'] = data['species'].map(target_mapping)
    print("✅ Dataset Loaded Successfully!")
    return data
