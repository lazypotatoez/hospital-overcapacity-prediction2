import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the dataset and model
data = pd.read_csv("Dataset_with_Overcapacity.csv")
model = load_model("roanne_fnn_model.h5")

# Define feature columns used during training
feature_columns = ["Admissions", "Hospital Admissions", "Number of Beds", "Admissions per Bed"]  # Adjust as needed
target_column = "Overcapacity"

# Preprocessing: Scale features
scaler = StandardScaler()
X = data[feature_columns]
y = data[target_column]
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train = X[data["Year"] < 2018]
X_test = X[data["Year"] >= 2018]
y_train = y[data["Year"] < 2018]
y_test = y[data["Year"] >= 2018]

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Streamlit app UI
st.title("Hospital Overcapacity Prediction")
year_input = st.number_input("Enter Year to Predict (e.g., 2025)", min_value=2006, max_value=2023, step=1)

st.write("Columns in dataset:")
st.json(list(data.columns))

if st.button("Predict for Selected Year"):
    if year_input not in data["Year"].values:
        st.error("Year not in dataset!")
    else:
        selected_year_data = data[data["Year"] == year_input][feature_columns]
        selected_year_scaled = scaler.transform(selected_year_data)
        predictions = model.predict(selected_year_scaled)
        st.write(f"Predictions for {year_input}:")
        st.write(predictions)

if st.button("Show Test Set Evaluation"):
    test_predictions = model.predict(X_test_scaled)
    st.write("Test Predictions vs Actuals:")
    st.write(pd.DataFrame({"Actual": y_test.values, "Predicted": test_predictions.flatten()}))
