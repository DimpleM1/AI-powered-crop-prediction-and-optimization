import streamlit as st
import pandas as pd
import joblib
import numpy as np
import urllib.request
import os

st.set_page_config(page_title="AI Crop Recommendation System", page_icon="🌱", layout="centered")

st.title("🌱 AI Powered Crop Recommendation System")
st.write("This automated interface utilizes a Random Forest Classifier to predict ideal crop cultivation paths.")

# 1. Automatically fetch the heavy model from Google Drive if it isn't downloaded yet
MODEL_URL = "https://drive.google.com/file/d/1v64ALYoJqtt3QWRziw343lW-u8WgijiZ/view?usp=drive_link"
MODEL_PATH = "crop_model.pkl"

@st.cache_resource
def load_core_engine():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Initializing AI Engine binaries from secure cloud storage..."):
            # This downloads the file seamlessly in the background
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    # Load all local components
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, scaler, label_encoder

try:
    model, scaler, label_encoder = load_core_engine()
    
    st.header("📊 Enter Environmental Metrics")
    temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    ph = st.number_input("Soil pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.number_input("Rainfall Volume (mm)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)

    if st.button("Predict Optimal Crop Variety"):
        features = np.array([[temp, humidity, ph, rainfall]])
        scaled_features = scaler.transform(features)
        
        # Predict and decode the name
        prediction_encoded = model.predict(scaled_features)
        crop_name = label_encoder.inverse_transform(prediction_encoded)
        
        st.balloons()
        st.success(f"🌾 The recommended crop variety for these conditions is: **{crop_name[0].upper()}**")

except Exception as e:
    st.error(f"Configuration sync error: Please ensure your cloud hosting permissions are active. Details: {e}")
