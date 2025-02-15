import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model("roanne_fnn_model.h5")

# Streamlit app
st.title("Hospital Overcapacity Prediction")

st.write("Enter feature values to predict overcapacity:")

# Input features
input_features = st.text_input("Enter feature values (comma-separated):")

# Predict button
if st.button("Predict"):
    try:
        # Process input
        features = np.array([float(x) for x in input_features.split(",")]).reshape(1, -1)
        prediction = model.predict(features)[0][0]
        prediction_label = "Overcapacity" if prediction > 0.5 else "Not Overcapacity"
        st.write(f"Prediction: {prediction_label}")
    except Exception as e:
        st.write("Error in processing input:", e)
