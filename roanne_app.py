import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model("roanne_fnn_model.h5")

# Load and scale data
data = pd.read_csv("Final_Merged_Dataset.csv")

# Dynamically drop columns that exist in the dataset
columns_to_drop = ["Year", "Overcapacity"]
columns_to_drop = [col for col in columns_to_drop if col in data.columns]

# Select only numeric columns for scaling
numeric_columns = data.drop(columns=columns_to_drop, axis=1).select_dtypes(include=[np.number]).columns

scaler = StandardScaler()
scaler.fit(data[numeric_columns])  # Fit scaler only on numeric columns

# Sidebar layout for navigation
st.sidebar.title("Hospital Overcapacity Prediction")
page = st.sidebar.radio("Go to", ["Predict", "Explore Dataset"])

if page == "Predict":
    # Prediction Page
    st.title("Hospital Overcapacity Prediction")

    st.write("Enter feature values to predict overcapacity:")

    # Input features
    input_features = st.text_input("Enter feature values (comma-separated):")
    try:
        # Predict button
        if st.button("Predict"):
            # Convert input features into a NumPy array
            features = np.array(input_features.split(","), dtype=float).reshape(1, -1)

            # Scale input features
            features_scaled = scaler.transform(features)

            # Make prediction
            prediction = model.predict(features_scaled)[0][0]

            # Display the prediction
            st.write(f"Predicted Overcapacity Probability: {prediction:.4f}")
            if prediction > 0.5:
                st.write("Prediction: Overcapacity (Yes)")
            else:
                st.write("Prediction: No Overcapacity")
    except Exception as e:
        st.write("Error in processing input:")
        st.error(e)

elif page == "Explore Dataset":
    # Dataset Exploration Page
    st.title("Explore the Dataset")
    st.write("View the dataset used for training:")
    st.dataframe(data)

    st.write("Summary statistics:")
    st.write(data.describe())
