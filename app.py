
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model and scaler
model = joblib.load("best_rf_advanced_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("💓 Heart Disease Risk Prediction App")

st.markdown("Provide the following details to assess heart disease risk.")

# 🧾 User Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
chest_pain_type = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
serum_cholesterol = st.number_input("Serum Cholesterol (mg/dl)", value=200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[0, 1])
restecg = st.selectbox("Resting ECG (0: Normal, 1: Abnormal, 2: Probable LVH)", options=[0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina?", options=[0, 1])
oldpeak = st.number_input("ST depression induced by exercise", value=1.0, format="%.1f")
slope = st.selectbox("Slope of ST Segment (0–2)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0–3)", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible)", options=[1, 2, 3])

# ⏩ Predict button
if st.button("Predict"):
    # 📦 Prepare input
    input_data = pd.DataFrame([[
        age, sex, chest_pain_type, resting_blood_pressure, serum_cholesterol,
        fasting_blood_sugar, restecg, thalach, exang, oldpeak, slope, ca, thal
    ]], columns=[
        'age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol',
        'fasting_blood_sugar', 'restecg', 'thalach', 'exang', 'oldpeak',
        'slope', 'ca', 'thal'
    ])

    # 🔄 Apply same scaling as training
    input_scaled = scaler.transform(input_data)

    # 🧠 Make prediction
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][prediction]

    # ✅ Show result
    if prediction == 1:
        st.error(f"🔴 **High risk of heart disease** (confidence: {proba:.2%})")
    else:
        st.success(f"🟢 **Low risk of heart disease** (confidence: {proba:.2%})")
