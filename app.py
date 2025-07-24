import streamlit as st
import pandas as pd
import joblib

# Load model and corrected scaler
model = joblib.load("final_heart_attack_model.pkl")
scaler = joblib.load("scaler.pkl")   # <-- this is the new corrected scaler

# Streamlit page config
st.set_page_config(page_title="Heart Attack Prediction", layout="centered")
st.title("🫀 Heart Attack Risk Prediction App")
st.markdown("""
This app predicts the risk of heart attack using a model trained on medical and network features.
Please enter the patient's clinical details below:
""")

# === Form for input ===
with st.form("input_form"):
    st.header("📋 Patient Clinical Details")

    patient_id = st.number_input("Patient ID (for your record only)", min_value=1, max_value=999999)
    age = st.number_input("Age (years)", min_value=1, max_value=120)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    systolic_bp = st.number_input("Systolic Blood Pressure (mm Hg)", min_value=80, max_value=250)
    diastolic_bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=50, max_value=150)
    chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                      format_func=lambda x: ["Typical Angina", "Atypical", "Non-anginal", "Asymptomatic"][x])
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    thalachh = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250)
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2])
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                           format_func=lambda x: ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][x])
    ca = st.selectbox("Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia Type", options=[1, 2, 3],
                        format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x])

    submitted = st.form_submit_button("🔍 Predict")

# === On Submit ===
if submitted:
    # Prepare dictionary (graph features default 0)
    input_dict = {
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
        'thal': thal,
        'pageRank': 0,
        'betweenness': 0,
        'community': 0,
        'degree': 0,
        'eigenvector': 0,
        'avgSimilarity': 0
    }

    # Match scaler column order (no PatientID, no target)
    required_order = [
        'age','sex','systolic_bp','diastolic_bp','chol','fbs','cp','exang',
        'thalachh','oldpeak','slope','restecg','ca','thal',
        'pageRank','betweenness','community','degree','eigenvector','avgSimilarity'
    ]
    input_df = pd.DataFrame([input_dict])[required_order]

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # Display result
    st.subheader("🧾 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Attack ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Attack ({(1 - prob)*100:.2f}%)")

    # Show entered data
    st.markdown("---")
    st.markdown("### 🔢 Entered Patient Data (graph features hidden by default)")
    st.dataframe(input_df.T, use_container_width=True)
