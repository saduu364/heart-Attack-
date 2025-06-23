
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_rf_advanced_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction App ❤️")
st.write("Enter patient details to predict the likelihood of heart disease.")

# Input fields
age = st.number_input("Age", 18, 100, 50)
sex = st.selectbox("Sex", [0, 1])
chest_pain_type = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
resting_blood_pressure = st.number_input("Resting Blood Pressure", 80, 200, 120)
serum_cholesterol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of ST segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (0 = Normal; 1 = Fixed Defect; 2 = Reversible Defect)", [0, 1, 2])

if st.button("Predict"):
    # Feature names must match training exactly
    feature_columns = [
        'age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
        'serum_cholesterol', 'fasting_blood_sugar', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

    input_data = pd.DataFrame([[
        age, sex, chest_pain_type, resting_blood_pressure,
        serum_cholesterol, fasting_blood_sugar, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]], columns=feature_columns)

    # Scale input
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][prediction]

    st.subheader("🔍 Prediction Result")
    st.write("Prediction:", "Heart Disease" if prediction == 1 else "No Heart Disease")
    st.write(f"Confidence: {proba:.2%}")
