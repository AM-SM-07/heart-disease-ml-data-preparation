import streamlit as st
import pandas as pd
import joblib

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("KNN_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.title("Heart stroke prediction by Amit🫀")
st.markdown("Provide the following details :")

age = st.slider("Age", 18,100,25)
sex = st.selectbox("SEX", ['M','F'])
ChestPain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
Cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
MaxHR = st.slider("Max Heart Rate", 60, 220, 150)
ExerciseAngina = st.selectbox("Excercise-Induced Angina", ["Y", "N"])
Oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["UP", "Flat", "Down"])

if st.button("Predict"):

    raw_input = {
        'Age': age,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'FastingBS': FastingBS,
        'MaxHR': MaxHR,
        'Oldpeak': Oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + ChestPain: 1,
        'RestingECG_' + RestingECG: 1,
        'ExerciseAngina_' + ExerciseAngina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Exact same column order as training
    input_df = input_df[expected_columns]

    # Scale
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    # Probability
    probability = model.predict_proba(scaled_input)[0][1]
    probability_percentage = probability * 100

    st.write("Prediction:", prediction)
    st.write(f"Heart Disease Probability: {probability:.2%}%")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")