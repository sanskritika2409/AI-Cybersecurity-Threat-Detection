import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model
from src.detect import detect_threats
from src.visualize import plot_confusion_matrix

print("🚀 Starting Project...")

# Step 1: Load data
print("📥 Loading data...")
data = pd.read_csv("data/cyber_data.csv")

print("Data Loaded Successfully!")
print(data.head())

# Step 2: Preprocess
X, y = preprocess_data(data)

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Train model
model = train_model(X_train, y_train)

# Step 5: Evaluate
y_pred = evaluate_model(model, X_test, y_test)

# Step 6: Detect threats
results = detect_threats(y_pred)

print("\n🔍 Sample Results:")
for r in results[:10]:
    print(r)

# Step 7: Visualization
plot_confusion_matrix(y_test, y_pred)
import joblib

joblib.dump(model, "models/cyber_model.pkl")
print("💾 Model saved successfully!")
print(data.head())