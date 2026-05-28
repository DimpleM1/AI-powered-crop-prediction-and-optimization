
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model artifacts
model = joblib.load('crop_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🌱 AI Crop Recommendation System")
st.write("Enter the environmental metrics below to predict the optimal crop:")

# Input fields
temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

if st.button("Predict Optimal Crop"):
    # Create feature array matching what the pipeline expects
    features = np.array([[temp, humidity, ph, rainfall]])
    prediction = model.predict(features)
    st.success(f"🌾 The recommended crop for these conditions is: **{prediction[0]}**")
