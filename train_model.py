
# CreditWise - Loan Approval Prediction
# Model Training

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
print("Loading dataset...")

df = pd.read_csv("dataset/loan_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Clean the dataset
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

# Remove records without a loan status
df = df.dropna(subset=["Loan_Status"])

# Convert target values into numbers
df["Loan_Status"] = (
    df["Loan_Status"]
    .astype(str)
    .str.strip()
    .map({"Y": 1, "N": 0, "Yes": 1, "No": 0})
)

df = df.dropna(subset=["Loan_Status"])
df["Loan_Status"] = df["Loan_Status"].astype(int)

# 3. Select features
# Exclude Applicant ID and Gender.
categorical_features = [
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

numerical_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

features = categorical_features + numerical_features

X = df[features].copy()
y = df["Loan_Status"]

# Clean categorical values
import numpy as np

for col in categorical_features:
    X[col] = X[col].apply(
        lambda value: str(value).strip()
        if pd.notna(value) else np.nan
    )

print("\nInput features:", features)
print("Target variable: Loan_Status")

# 4. Preprocess numerical and categorical data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 5. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Create the Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# 7. Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)
print("Model training completed!")

# 8. Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation")
print("-------------------------")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Save the trained model
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/loan_model.pkl")

print("\nModel saved successfully!")
print("Location: model/loan_model.pkl")
print("\nCreditWise model is ready!")