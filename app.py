import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("hybrid_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🫀 Heart Attack Risk Prediction App")
st.markdown("This app predicts the risk of heart attack using hybrid graph and clinical features.")

st.header("📋 Enter Patient Clinical Information:")

# Input fields
age = st.number_input("Age (years)", 20, 100, 55)
sex = st.selectbox("Sex", ["Male", "Female"])
systolic_bp = st.number_input("Systolic Blood Pressure (mm Hg)", 80, 250, 120)
diastolic_bp = st.number_input("Diastolic Blood Pressure (mm Hg)", 60, 200, 80)
chol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
thalachh = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
ca = st.selectbox("Major Vessels Colored", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# Encode inputs manually (same as in dataset)
sex_val = 1 if sex == "Male" else 0
fbs_val = 1 if fbs == "Yes" else 0
cp_val = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal": 2, "Asymptomatic": 3}[cp]
exang_val = 1 if exang == "Yes" else 0
slope_val = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
restecg_val = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}[restecg]
thal_val = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}[thal]

# Create input array
input_data = np.array([[sex_val, restecg_val, ca, systolic_bp, thal_val,
                        fbs_val, cp_val, thalachh, slope_val, age, chol, oldpeak]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("🔍 Predict"):
    prob = model.predict_proba(input_scaled)[0][1]
    label = "High" if prob >= 0.5 else "Low"
    st.subheader("🧾 Prediction Result:")
    st.success(f"✅ {label} Risk of Heart Attack ({prob*100:.2f}% probability)")
    
    st.markdown("### 🔎 Entered Data")
    st.json({
        "Age": age, "Sex": sex, "Systolic BP": systolic_bp, "Diastolic BP": diastolic_bp,
        "Cholesterol": chol, "FBS >120": fbs, "Chest Pain": cp, "Exang": exang,
        "Max Heart Rate": thalachh, "Oldpeak": oldpeak, "Slope": slope,
        "Rest ECG": restecg, "Vessels Colored": ca, "Thalassemia": thal
    })
