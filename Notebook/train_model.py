import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Load Dataset
# Ensure your dataset is downloaded into the Dataset folder
DATA_PATH = os.path.join("..", "Dataset", "loan_data.csv")

# If running directly from root directory:
if not os.path.exists(DATA_PATH):
    DATA_PATH = "Dataset/loan_data.csv"

df = pd.read_csv(DATA_PATH)

# Drop Loan_ID if present as it has no predictive power
if "Loan_ID" in df.columns:
    df.drop(columns=["Loan_ID"], inplace=True)

# 2. Encode Target Variable ('Y' -> 1, 'N' -> 0)
target_column = "Loan_Status"
df[target_column] = df[target_column].map({"Y": 1, "N": 0})
df.dropna(subset=[target_column], inplace=True)

X = df.drop(columns=[target_column])
y = df[target_column]

# 3. Feature Engineering: Total Income
if "ApplicantIncome" in X.columns and "CoapplicantIncome" in X.columns:
    X["TotalIncome"] = X["ApplicantIncome"] + X["CoapplicantIncome"]

# Identify column types
numeric_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "TotalIncome"]
numeric_features = [col for col in numeric_features if col in X.columns]

categorical_features = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Credit_History", "Property_Area"]
categorical_features = [col for col in categorical_features if col in X.columns]

# Ensure Credit_History is treated as categorical/discrete string
if "Credit_History" in X.columns:
    X["Credit_History"] = X["Credit_History"].astype(str)

# 4. Preprocessing Pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 6. Model Training & Comparison (Requirement: At least 3 models)
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

best_model = None
best_f1 = -1.0
best_model_name = ""

print("===== Model Evaluation Results =====")
for name, clf in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print(f"\nModel: {name}")
    print(f"Accuracy:  {acc:.4f} | Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f} | F1-Score:  {f1:.4f} | ROC-AUC: {roc:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = name

print(f"\nBest Performing Model Selected: {best_model_name}")

# 7. Save Model Artifact
os.makedirs("Model", exist_ok=True)
joblib.dump(best_model, "Model/loan_approval_model.pkl")
print("Model pipeline successfully saved to 'Model/loan_approval_model.pkl'")