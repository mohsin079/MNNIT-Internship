import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="centered"
)

# Load the trained model pipeline
@st.cache_resource
def load_model():
    model_path = os.path.join("Model", "loan_approval_model.pkl")
    if not os.path.exists(model_path):
        model_path = "loan_approval_model.pkl"
    return joblib.load(model_path)

st.title("🏦 Loan Approval Prediction System")
st.write("Fill in the applicant details below to evaluate eligibility.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'train_model.py' has been executed.")
    st.stop()

# Input Form
with st.form("loan_application_form"):
    st.subheader("Applicant Information")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])

    with col2:
        applicant_income = st.number_input("Applicant Income ($/mo)", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Coapplicant Income ($/mo)", min_value=0, value=0, step=500)
        loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, value=150, step=10)
        loan_amount_term = st.selectbox("Loan Term (Months)", [360, 180, 120, 240, 300, 480])
        credit_history = st.selectbox("Credit History", ["1.0", "0.0"], format_func=lambda x: "Good (1.0)" if x == "1.0" else "Bad / No History (0.0)")
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Predict Loan Status")

if submitted:
    # Compute Engineered Feature
    total_income = applicant_income + coapplicant_income

    # Match feature names used in training
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": str(credit_history),
        "Property_Area": property_area,
        "TotalIncome": total_income
    }])

    # Generate Prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.markdown("---")
    if prediction == 1:
        st.success(f"**Status: Approved** (Approval Probability: {probabilities[1]:.2%})")
    else:
        st.error(f"**Status: Rejected** (Approval Probability: {probabilities[1]:.2%})")