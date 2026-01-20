import streamlit as st
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Breast Cancer Diagnostic System", page_icon="🎗️")

# Load the saved model and scaler from the /model/ directory
try:
    classifier = joblib.load('model/breast_cancer_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
except Exception as e:
    st.error(f"Error loading model files. Please ensure they are in the '/model/' folder. Details: {e}")

st.title("Breast Cancer Prediction System")
st.markdown("---")

# Part B Requirement: User Inputs for tumor features
st.sidebar.header("Tumor Feature Inputs")
st.sidebar.info("Please enter the mean values calculated from the FNA image.")

# The 5 features selected in the notebook
radius = st.sidebar.number_input("Mean Radius", min_value=0.0, format="%.3f")
texture = st.sidebar.number_input("Mean Texture", min_value=0.0, format="%.3f")
perimeter = st.sidebar.number_input("Mean Perimeter", min_value=0.0, format="%.3f")
area = st.sidebar.number_input("Mean Area", min_value=0.0, format="%.3f")
smoothness = st.sidebar.number_input("Mean Smoothness", min_value=0.0, format="%.5f")

# Prediction logic
if st.button("Run Prediction"):
    # Arrange inputs into the same order as training
    input_data = np.array([[radius, texture, perimeter, area, smoothness]])
    
    # Scale the inputs using the saved scaler (Mandatory for distance-based models)
    scaled_data = scaler.transform(input_data)
    
    # Generate prediction
    prediction = classifier.predict(scaled_data)
    
    # Display Result
    if prediction[0] == 0:
        st.error("### Prediction Result: **Malignant**")
        st.write("The tumor is classified as cancerous.")
    else:
        st.success("### Prediction Result: **Benign**")
        st.write("The tumor is classified as non-cancerous.")

# Part A Requirement: Educational disclaimer
st.markdown("---")
st.caption("**Disclaimer:** This system is strictly for educational purposes and must not be presented as a medical diagnostic tool.")