import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="AI Crop Recommendation System", page_icon="🌱", layout="centered")

st.title("🌱 AI Powered Crop Recommendation System")
st.write("This interactive interface utilizes your trained machine learning model to predict ideal cultivation paths.")

# Sidebar for Model Artifact Uploads to completely bypass GitHub size limits
st.sidebar.header("📁 1. Load Your Models")
st.sidebar.info("Upload your local files below to turn on the prediction engine.")
uploaded_model = st.sidebar.file_uploader("Upload 'crop_model.pkl'", type=["pkl"])
uploaded_scaler = st.sidebar.file_uploader("Upload 'scaler.pkl'", type=["pkl"])

if uploaded_model is not None and uploaded_scaler is not None:
    # Safely load the binary artifacts directly from browser memory
    model = joblib.load(uploaded_model)
    scaler = joblib.load(uploaded_scaler)
    st.sidebar.success("✅ Engine binaries activated successfully!")
    
    st.header("📊 2. Enter Environmental Metrics")
    temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    ph = st.number_input("Soil pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.number_input("Rainfall Volume (mm)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)

    if st.button("Predict Optimal Crop Variety"):
        features = np.array([[temp, humidity, ph, rainfall]])
        scaled_features = scaler.transform(features)
        
        # Make the prediction
        prediction = model.predict(scaled_features)
        
        st.balloons()
        st.success(f"🌾 The recommended crop variety for these conditions is: **{prediction[0].upper()}**")
else:
    st.warning("👈 Missing Binaries: Please drop your 'crop_model.pkl' and 'scaler.pkl' into the sidebar upload zones to launch the interface functionality.")
