# app.py
import streamlit as st
import pandas as pd
import joblib

# --- Load model and scaler ---
model = joblib.load("final_heart_attack_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Page Config ---
st.set_page_config(page_title="Heart Attack Predictor", page_icon="🫀")
st.title("🫀 Heart Attack Risk Prediction App")
st.markdown("This app predicts heart attack risk using clinical features and graph intelligence (hidden).")

# --- User Inputs ---
with st.form("input_form"):
    st.subheader("Enter Patient Clinical Information:")
    age = st.number_input("Age (years)", min_value=1, max_value=120)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    systolic_bp = st.number_input("Systolic Blood Pressure (mm Hg)", min_value=80, max_value=250)
    diastolic_bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=50, max_value=150)
    chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    thalachh = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250)
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x])

    submitted = st.form_submit_button("🔍 Predict")

# --- Prediction ---
if submitted:
    clinical_input = pd.DataFrame([{
        'age': age, 'sex': sex, 'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
        'chol': chol, 'fbs': fbs, 'cp': cp, 'exang': exang, 'thalachh': thalachh,
        'oldpeak': oldpeak, 'slope': slope, 'restecg': restecg, 'ca': ca, 'thal': thal,

        # Hidden GDS features - Default average values
        'pageRank': 0.15,
        'betweenness': 0.10,
        'community': 3,
        'degree': 0.22,
        'eigenvector': 0.12,
        'avgSimilarity': 0.08
    }])

    # Scale
    input_scaled = scaler.transform(clinical_input)

    # Predict
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("🧾 Prediction Result:")
    if pred == 1:
        st.error(f"⚠️ High Risk of Heart Attack ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Attack ({(1 - prob)*100:.2f}%)")

    with st.expander("🔎 Entered Data"):
        st.dataframe(clinical_input.drop(columns=[
            'pageRank', 'betweenness', 'community', 'degree', 'eigenvector', 'avgSimilarity'
        ]).T, use_container_width=True)
