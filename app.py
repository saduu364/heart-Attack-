# app.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler (clinical only)
model = joblib.load("clinical_only_model.pkl")
scaler = joblib.load("clinical_only_scaler.pkl")

# App Title
st.title("🫀 Heart Attack Risk Prediction App")
st.markdown("""
This app predicts the risk of heart attack using **clinical features**.
Enter patient data below and click **Predict**.
""")

# Form for user input
with st.form("input_form"):
    st.subheader("📋 Enter Patient Clinical Information:")

    age = st.number_input("Age (years)", min_value=1, max_value=120)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    systolic_bp = st.number_input("Systolic Blood Pressure (mm Hg)", min_value=80, max_value=250)
    diastolic_bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=40, max_value=150)
    chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: [
        "Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    thalachh = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250)
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", options=[0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2], format_func=lambda x: [
        "Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x])
    ca = st.selectbox("Major Vessels Colored", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[1, 2, 3], format_func=lambda x: {
        1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x])

    submitted = st.form_submit_button("🔍 Predict")

# Prediction
if submitted:
    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'chol': chol,
        'fbs': fbs,
        'cp': cp,
        'exang': exang,
        'thalachh': thalachh,
        'oldpeak': oldpeak,
        'slope': slope,
        'restecg': restecg,
        'ca': ca,
        'thal': thal
    }])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("🧾 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Attack ({prob*100:.2f}% probability)")
    else:
        st.success(f"✅ Low Risk of Heart Attack ({(1 - prob)*100:.2f}% probability)")

    st.markdown("🔎 **Entered Data**")
    st.dataframe(input_data.T)

