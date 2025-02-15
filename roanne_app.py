import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load dataset and model
data = pd.read_csv("Dataset_with_Overcapacity.csv")
model = load_model("roanne_fnn_model.h5")

# Define columns to drop and target column
columns_to_drop = ["Year", "Admission Type", "Service Type", "Facility Type"]
target_column = "Overcapacity"

# Remove non-numeric columns and prepare features/labels
X = data.drop(columns=columns_to_drop + [target_column], axis=1)
y = data[target_column].reset_index(drop=True)

# Split into train/test sets
train_data = data[data["Year"] < 2018]
test_data = data[data["Year"] >= 2018]

X_train = train_data.drop(columns=columns_to_drop + [target_column], axis=1)
y_train = train_data[target_column]

X_test = test_data.drop(columns=columns_to_drop + [target_column], axis=1)
y_test = test_data[target_column]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Streamlit app
st.title("Hospital Overcapacity Prediction")
st.sidebar.title("Hospital Overcapacity Prediction")

# Display dataset columns
st.sidebar.subheader("Columns in dataset:")
st.sidebar.write(data.columns.tolist())

# Input for year prediction
year_input = st.number_input("Enter Year to Predict (e.g., 2025)", min_value=2006, max_value=2023, step=1)
if st.button("Predict for Selected Year"):
    try:
        # Filter data for the selected year
        year_data = data[data["Year"] == year_input]
        if year_data.empty:
            st.error("No data available for the selected year.")
        else:
            # Prepare features for the model
            year_features = year_data.drop(columns=columns_to_drop + [target_column], axis=1)
            year_scaled = scaler.transform(year_features)

            # Make predictions
            predictions = model.predict(year_scaled)
            year_data["Predicted Overcapacity"] = predictions

            # Show predictions
            st.subheader("Predictions for the Selected Year")
            st.write(year_data[["Year", "Predicted Overcapacity"]])
    except Exception as e:
        st.error(f"Error in processing input: {e}")

# Button to show test set evaluation
if st.button("Show Test Set Evaluation"):
    try:
        # Evaluate on test data
        test_predictions = model.predict(X_test_scaled)
        test_results = pd.DataFrame({"Actual": y_test.values, "Predicted": test_predictions.flatten()})
        
        # Display results
        st.subheader("Test Set Evaluation Results")
        st.write(test_results.head(10))
    except Exception as e:
        st.error(f"Error during test set evaluation: {e}")
