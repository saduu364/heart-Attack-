import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("hybrid_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🫀 Heart Attack Risk Prediction App")
st.markdown("This app predicts the risk of heart attack using **hybrid graph and clinical features**.")

st.header("📋 Enter Patient Clinical Information:")

# Clinical Inputs
age = st.number_input("Age (years)", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
sex_val = 1 if sex == "Male" else 0

systolic_bp = st.number_input("Systolic Blood Pressure (mm Hg)", 80, 250, 120)
diastolic_bp = st.number_input("Diastolic Blood Pressure (mm Hg)", 50, 150, 80)

chol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])
fbs_val = 1 if fbs == "Yes" else 0

cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"])
cp_val = ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"].index(cp)

exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
exang_val = 1 if exang == "Yes" else 0

thalachh = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)

slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)

restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
restecg_val = ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(restecg)

ca = st.slider("Major Vessels Colored", 0, 4, 0)

thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
thal_val = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)

# Graph-based Features - Use defaults or allow user to tweak
st.subheader("📊 Graph Intelligence Features (Optional, use defaults if unsure)")

pageRank = st.number_input("PageRank", 0.0, 1.0, 0.12)
betweenness = st.number_input("Betweenness Centrality", 0.0, 1.0, 0.08)
community = st.number_input("Community ID", 0, 10, 2)
degree = st.number_input("Node Degree", 0, 20, 4)
eigenvector = st.number_input("Eigenvector Centrality", 0.0, 1.0, 0.09)
avgSimilarity = st.number_input("Average Node Similarity", 0.0, 1.0, 0.16)

if st.button("🔮 Predict"):
    input_data = np.array([[
        age, sex_val, systolic_bp, diastolic_bp, chol, fbs_val, cp_val, exang_val,
        thalachh, oldpeak, slope_val, restecg_val, ca, thal_val,
        pageRank, betweenness, community, degree, eigenvector, avgSimilarity
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("🧾 Prediction Result:")
    risk_label = "✅ Low Risk of Heart Attack" if prediction == 0 else "⚠️ High Risk of Heart Attack"
    st.success(f"{risk_label} ({probability * 100:.2f}% probability)")

    st.subheader("🔎 Entered Data")
    st.write(dict(
        age=age, sex=sex, systolic_bp=systolic_bp, diastolic_bp=diastolic_bp,
        chol=chol, fbs=fbs, cp=cp, exang=exang, thalachh=thalachh, oldpeak=oldpeak,
        slope=slope, restecg=restecg, ca=ca, thal=thal,
        pageRank=pageRank, betweenness=betweenness, community=community,
        degree=degree, eigenvector=eigenvector, avgSimilarity=avgSimilarity
    ))
