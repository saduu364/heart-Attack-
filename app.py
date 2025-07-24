# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load("final_heart_attack_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🫀 Heart Attack Risk Prediction App")
st.markdown("""
This app uses a Random Forest model trained on hybrid clinical-graph features to predict the likelihood of a heart attack.
Please enter the patient's details below:
""")

# Form
with st.form("input_form"):
    patient_id = st.text_input("Patient ID (for your record only)")

    age = st.number_input("Age (years)", min_value=1, max_value=120)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    systolic_bp = st.number_input("Systolic Blood Pressure (mm Hg)", min_value=80, max_value=250)
    diastolic_bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=50, max_value=150)
    chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    cp = st.selectbox("Chest Pain Type (0–Typical Angina, 1–Atypical, 2–Non-anginal, 3–Asymptomatic)", options=[0,1,2,3])
    exang = st.selectbox("Exercise Induced Angina", options=[0,1], format_func=lambda x: "No" if x == 0 else "Yes")
    thalachh = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250)
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0,1,2])
    restecg = st.selectbox("Resting ECG (0–Normal, 1–ST-T, 2–LVH)", options=[0,1,2])
    ca = st.selectbox("Major Vessels Colored by Fluoroscopy", options=[0,1,2,3])
    thal = st.selectbox("Thalassemia Type", options=[1,2,3], format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
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

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("🧾 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Attack ({prob*100:.2f}% probability)")
    else:
        st.success(f"✅ Low Risk of Heart Attack ({(1 - prob)*100:.2f}% probability)")

    st.markdown("---")
    st.markdown("🔢 **Entered Patient Data:**")
    st.dataframe(input_df.T, use_container_width=True)
