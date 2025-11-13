import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os
from huggingface_hub import login, HfApi

# Download and load the predictive maintenance model
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("HF_TOKEN is not set. Cannot download model.")
else:
    try:
        model_path = hf_hub_download(repo_id="sudha1726/predictive_maintainance_model", filename="best_predictive_maintainance_model_v1.joblib", token=HF_TOKEN)
        model = joblib.load(model_path)
        st.success("Predictive Maintenance Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        model = None

# Streamlit UI for Predictive Maintenance
st.title("Engine Predictive Maintenance App")
st.write("""
This application predicts whether an engine requires maintenance based on real-time sensor data.
Please enter the engine sensor readings below to get a prediction.
""")

if model:
    # User input fields for engine sensor data
    engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=3000, value=700)
    lub_oil_pressure = st.number_input("Lub Oil Pressure (bar/kPa)", min_value=0.0, max_value=10.0, value=3.0, format="%.2f")
    fuel_pressure = st.number_input("Fuel Pressure (bar/kPa)", min_value=0.0, max_value=30.0, value=7.0, format="%.2f")
    coolant_pressure = st.number_input("Coolant Pressure (bar/kPa)", min_value=0.0, max_value=10.0, value=2.5, format="%.2f")
    lub_oil_temp = st.number_input("Lub Oil Temperature (°C)", min_value=0.0, max_value=100.0, value=75.0, format="%.2f")
    coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=0.0, max_value=200.0, value=80.0, format="%.2f")

    # Assemble input data into DataFrame
    input_data = pd.DataFrame([{
        'Engine rpm': engine_rpm,
        'Lub oil pressure': lub_oil_pressure,
        'Fuel pressure': fuel_pressure,
        'Coolant pressure': coolant_pressure,
        'lub oil temp': lub_oil_temp,
        'Coolant temp': coolant_temp
    }])

    # Prediction
    if st.button("Predict Engine Condition"):
        prediction = model.predict(input_data)[0]
        result = "requires maintenance (Faulty)" if prediction == 1 else "does not require maintenance (Normal)"
        st.subheader("Prediction Result:")
        st.success(f"The model predicts the engine **{result}**")

    st.write("---")
st.subheader("Upload Deployment to Hugging Face Space")

if st.button("Upload Deployment Files"):
    if not HF_TOKEN:
        st.error("HF_TOKEN is not set. Cannot upload files.")
    else:
        login(token=HF_TOKEN)
        api = HfApi()
        repo_id = "sudha1726/predictive-maintainanace"  # your Space repo
        folder_path = "predictive_maintainance_project/deployment"

        try:
            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                repo_type="space",
                path_in_repo=""
            )
            st.success("Deployment files uploaded successfully to Hugging Face Space!")
        except Exception as e:
            st.error(f"Error uploading deployment files: {e}")

