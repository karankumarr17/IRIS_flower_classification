from sklearn.model_selection import train_test_split

def preprocess_data(data):
    """Split data into training and testing sets."""
    X = data.drop(columns=['species'])
    y = data['species']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("✅ Data Preprocessing Completed!")
    return X_train, X_test, y_train, y_test
