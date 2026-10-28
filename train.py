import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Pima Indians Diabetes dataset (768 patients)
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# 2. Pick the 5 input features and the target (1 = diabetic, 0 = not)
features = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
X = df[features]
y = df["Outcome"]

# 3. Hold back 20% of the data to test the model on patients it hasn't seen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate it
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
print(classification_report(y_test, preds))

# 6. Save the trained model so the API can load it
joblib.dump(model, "diabetes_model.pkl")
print("Model saved as diabetes_model.pkl")