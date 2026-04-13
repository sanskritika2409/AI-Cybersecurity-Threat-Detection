import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(data):
    print("🧹 Cleaning data...")

    # Remove missing values
    data = data.dropna()

    # Convert text columns → numbers
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])

    # Split features and label
    X = data.drop("label", axis=1)
    y = data["label"]

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("✅ Data ready!")

    return X_scaled, y