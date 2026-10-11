import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

print("🚀 Step 1: Loading Dataset...")
df = pd.read_csv("creditcard.csv") 

print(f"Dataset Shape: {df.shape}")
print(f"Class Distribution Before SMOTE:\n{df['Class'].value_counts()}")

X = df.drop(columns=['Class'])
y = df['Class']

print("\n🔄 Step 2: Preprocessing & Scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("\n⚖️ Step 3: Applying SMOTE for Class Imbalance...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\n🌲 Step 4: Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_res, y_train_res)

print("\n📊 Step 5: Evaluating Model Performance...")
y_pred = rf_model.predict(X_test)

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred):.4f}")

print("\n💾 Step 6: Saving Model & Scaler for Deployment...")
joblib.dump(rf_model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.joblib')
print("✅ Day 1 Target Achieved! 'fraud_model.pkl' and 'scaler.joblib' are ready.")