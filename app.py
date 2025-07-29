import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================
# Load Model and Scaler
# ============================
with open("best_model_rf.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ============================
# App Title
# ============================
st.title("❤️ Heart Attack Risk Prediction")
st.write("Enter the patient's clinical data below to predict the risk of heart attack.")

# ============================
# Input Fields (Clinical Features Only)
# ============================
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
thalachh = st.number_input("Max Heart Rate (thalachh)", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise-Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, format="%.1f")
slope = st.selectbox("Slope of ST segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", [1, 2, 3])

# ============================
# Prepare Input for Prediction
# ============================
if st.button("Predict"):
    # Create DataFrame
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg, thalachh,
        exang, oldpeak, slope, ca, thal
    ]], columns=["age", "sex", "cp", "trestbps", "chol", "fbs",
                 "restecg", "thalachh", "exang", "oldpeak", "slope",
                 "ca", "thal"])

    # Scale input using saved scaler
    scaled_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # Display Result
    if prediction == 1:
        st.error(f"🚨 High Risk of Heart Attack! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk of Heart Attack. (Probability: {probability:.2%})")