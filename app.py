# app.py

import streamlit as st
import pickle
import numpy as np

# Load model, scaler, and feature names
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Clinical feature names in correct order
clinical_features = [
    'age', 'sex', 'systolic_bp', 'diastolic_bp', 'chol',
    'fbs', 'cp', 'exang', 'thalachh', 'oldpeak',
    'slope', 'restecg', 'ca', 'thal'
]

st.set_page_config(page_title="Heart Attack Predictor", layout="centered")
st.title("❤️ Heart Attack Risk Predictor")

st.markdown("Enter the patient’s clinical data below:")

# Input fields
user_input = {}
user_input['age'] = st.number_input("Age", 20, 100, 50)
user_input['sex'] = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
user_input['systolic_bp'] = st.number_input("Systolic BP (e.g. 120)", 80, 250, 120)
user_input['diastolic_bp'] = st.number_input("Diastolic BP (e.g. 80)", 50, 150, 80)
user_input['chol'] = st.number_input("Cholesterol", 100, 600, 200)
user_input['fbs'] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
user_input['cp'] = st.selectbox("Chest Pain Type (0=typical, 3=asymptomatic)", [0, 1, 2, 3])
user_input['exang'] = st.selectbox("Exercise-Induced Angina", [0, 1])
user_input['thalachh'] = st.number_input("Max Heart Rate Achieved", 70, 250, 150)
user_input['oldpeak'] = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
user_input['slope'] = st.selectbox("Slope of ST Segment", [0, 1, 2])
user_input['restecg'] = st.selectbox("Resting ECG", [0, 1, 2])
user_input['ca'] = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
user_input['thal'] = st.selectbox("Thalassemia (1=normal, 3=fixed defect, 6=reversible)", [1, 3, 6])

# Predict
if st.button("Predict"):
    input_array = np.array([user_input[feature] for feature in clinical_features]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🔴 High Risk of Heart Attack! (Probability: {prob:.2%})")
    else:
        st.success(f"🟢 Low Risk of Heart Attack (Probability: {prob:.2%})")

    st.markdown("---")
    st.subheader("Prediction Details")
    st.json(user_input)
