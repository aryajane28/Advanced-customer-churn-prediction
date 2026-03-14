import streamlit as st
import pandas as pd
import joblib

model = joblib.load("../models/churn_model.pkl")

st.title("Customer Churn Prediction App")

tenure = st.slider("Tenure (months)",1,72)
monthly = st.number_input("Monthly Charges")

data = pd.DataFrame({
    "tenure":[tenure],
    "MonthlyCharges":[monthly]
})

if st.button("Predict"):
    pred = model.predict(data)
    st.write("Prediction:", pred)