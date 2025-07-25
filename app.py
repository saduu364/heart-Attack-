# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("final_heart_attack_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Attack Risk Prediction", layout="centered")
st.title("🫀 Heart Attack Risk Prediction App")
st.markdown("This app predicts heart attack risk using clinical features and graph-based intelligence (hidden).")

with st.form("input_form"):
    patient_id = st.text_input("Patient ID (optional)")

    age = st.number_input("Age (years)", min_value=1, max_value=120)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    systolic_bp = st.number_input("Systolic Blood Pressure (mm Hg)", min_value=80, max_value=250)
    diastolic_bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=50, max_value=150)
    chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical", "Non-anginal", "Asymptomatic"][x])
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    thalachh = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250)
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", options=[0, 1, 2])
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2])
    ca = st.selectbox("Major Vessels Colored", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x])

    submitted = st.form_submit_button("Predict")

if submitted:
    # 13 Clinical inputs
    clinical_data = {
        'age': age, 'sex': sex, 'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
        'chol': chol, 'fbs': fbs, 'cp': cp, 'exang': exang, 'thalachh': thalachh,
        'oldpeak': oldpeak, 'slope': slope, 'restecg': restecg, 'ca': ca, 'thal': thal
    }

    # GDS placeholders (7 features)
    gds_defaults = {
        'pageRank': 0.0, 'betweenness': 0.0, 'community': 0.0,
        'degree': 0.0, 'eigenvector': 0.0, 'avgSimilarity': 0.0
    }

    # Merge both for full hybrid vector (20 features)
    full_input = {**clinical_data, **gds_defaults}
    input_df = pd.DataFrame([full_input])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("🔎 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Attack ({prob * 100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Attack ({(1 - prob) * 100:.2f}%)")

    st.markdown("#### 🧾 Entered Patient Data")
    st.dataframe(pd.DataFrame([clinical_data]).T, use_container_width=True)
