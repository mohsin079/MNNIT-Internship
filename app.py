import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide"
)
@st.cache_resource 
def load_artifacts():
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"⚠️ Error loading model files. Make sure 'fraud_model.pkl' and 'scaler.joblib' are in the same folder. Error: {e}")

st.title("💳 AI-Powered Credit Card Fraud Detection")
st.markdown("This system uses a **Random Forest Classifier** trained with **SMOTE** to detect fraudulent credit card transactions in real-time.")
st.divider() 

st.subheader("🕵️‍♂️ Enter Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    time = st.number_input("Transaction Time (Seconds from first transaction):", min_value=0.0, value=0.0, step=1.0)
    v1 = st.number_input("V1 Component:", value=0.0)
    v2 = st.number_input("V2 Component:", value=0.0)
    v3 = st.number_input("V3 Component:", value=0.0)
    v4 = st.number_input("V4 Component:", value=0.0)
    v5 = st.number_input("V5 Component:", value=0.0)
    v6 = st.number_input("V6 Component:", value=0.0)
    v7 = st.number_input("V7 Component:", value=0.0)
    v8 = st.number_input("V8 Component:", value=0.0)
    v9 = st.number_input("V9 Component:", value=0.0)

with col2:
    amount = st.number_input("Transaction Amount ($):", min_value=0.0, value=10.0, step=0.1)
    v10 = st.number_input("V10 Component:", value=0.0)
    v11 = st.number_input("V11 Component:", value=0.0)
    v12 = st.number_input("V12 Component:", value=0.0)
    v13 = st.number_input("V13 Component:", value=0.0)
    v14 = st.number_input("V14 Component:", value=0.0)
    v15 = st.number_input("V15 Component:", value=0.0)
    v16 = st.number_input("V16 Component:", value=0.0)
    v17 = st.number_input("V17 Component:", value=0.0)
    v18 = st.number_input("V18 Component:", value=0.0)

with col3:
    v19 = st.number_input("V19 Component:", value=0.0)
    v20 = st.number_input("V20 Component:", value=0.0)
    v21 = st.number_input("V21 Component:", value=0.0)
    v22 = st.number_input("V22 Component:", value=0.0)
    v23 = st.number_input("V23 Component:", value=0.0)
    v24 = st.number_input("V24 Component:", value=0.0)
    v25 = st.number_input("V25 Component:", value=0.0)
    v26 = st.number_input("V26 Component:", value=0.0)
    v27 = st.number_input("V27 Component:", value=0.0)
    v28 = st.number_input("V28 Component:", value=0.0)

st.divider()

if st.button("🔍 Analyze Transaction", type="primary"):
    input_data = np.array([[time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, 
                            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, 
                            v21, v22, v23, v24, v25, v26, v27, v28, amount]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    st.subheader("📊 Analysis Result")
    
    if prediction[0] == 1:
        st.error(f"🚨 **ALERT: High Risk of Fraud Detected!** (Confidence: {prediction_proba[0][1]*100:.2f}%)")
        st.warning("Recommendation: Block this transaction immediately and notify the customer.")
    else:
        st.success(f"🟢 **SUCCESS: Transaction is Safe & Legitimate.** (Confidence: {prediction_proba[0][0]*100:.2f}%)")
        st.info("Recommendation: Proceed with transaction processing.")