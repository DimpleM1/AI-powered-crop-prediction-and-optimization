# 🌱 AI-Powered Crop Recommendation System

A cloud-deployed Machine Learning application built to predict the most optimal crop varieties for cultivation based on dynamic environmental metrics. This project was developed as part of my Data Visualization and Machine Learning engineering internship.

🔗 **Live Application Link:** [Launch Web App Tool](https://ai-powered-crop-prediction-and-optimization-nkp6kmicstezvc67bg.streamlit.app/)

---

## 🚀 How to Evaluate the App (For Recruiters & Reviewers)

Because GitHub has strict file size limitations for machine learning model binaries (`.pkl`), this application utilizes a secure browser-side memory upload stream to execute live predictions. 

To run a prediction cycle:
1. Open the **Live Application Link** above.
2. Download the `crop_model.pkl` and `scaler.pkl` files from this repository to your local computer.
3. Drag and drop both **`crop_model.pkl`** and **`scaler.pkl`** into the dedicated upload zones in the web application's left sidebar.
4. Input your custom metrics for Temperature, Humidity, Soil pH, and Rainfall.
5. Click **"Predict Optimal Crop Variety"** to see the machine learning classification architecture compute the results with floating animations.

---

## 🛠️ Technology Stack & Architecture

* **Frontend Dashboard:** Streamlit (Python Core Framework)
* **Machine Learning Pipeline:** Scikit-Learn, Random Forest Classifier Architecture
* **Data Processing & Scaffolding:** Pandas, NumPy, Joblib Binary Serializers
* **Deployment & Hosting Core:** Streamlit Cloud Network Engine

---

## 📊 Core Features Included
* **Dynamic Feature Scaling:** Implements a pre-trained `StandardScaler` to clean incoming live telemetry metrics before prediction.
* **Label Encoding Transformation:** Automatically translates target integer predictions back into human-readable crop varieties using a saved `LabelEncoder` instance.
* **Responsive Sidebar UI:** Clean, sandboxed file processing environment that runs entirely in memory without tracking data footprint.
