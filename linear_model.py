import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Load the trained model
# ------------------------------
try:
    model = joblib.load("logistic_model.pkl")
except Exception as e:
    st.error(f"⚠️ Could not load the model: {e}")
    st.stop()

st.title("Loan Status Prediction App")
st.write("This app predicts **Loan Status** using a trained Logistic Regression model.")

# ------------------------------
# Option 1: Upload CSV for batch prediction
# ------------------------------
st.header("📂 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload your CSV file with features", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

    # Predict
    predictions = model.predict(data)
    data["Predicted Loan Status"] = predictions
    st.write("✅ Predictions:")
    st.dataframe(data)

    # Download predictions
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )

# ------------------------------
# Option 2: Single prediction via form
# ------------------------------
st.header("📝 Single Loan Application Prediction")

with st.form("prediction_form"):
    # NOTE: Replace with actual feature names from your dataset
    # For now I’ll assume some generic financial features
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount", min_value=0)
    Loan_Amount_Term = st.number_input("Loan Amount Term (months)", min_value=0)
    Credit_History = st.selectbox("Credit History", [0, 1])

    submitted = st.form_submit_button("Predict Loan Status")

    if submitted:
        input_data = pd.DataFrame(
            [[ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History]],
            columns=["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
        )

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Loan Status: **{prediction}**")

