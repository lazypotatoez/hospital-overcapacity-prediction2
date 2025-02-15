import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load dataset
data = pd.read_csv("Dataset_with_Overcapacity.csv")
st.write("Columns in dataset:", list(data.columns))

# Define target column and columns to drop
target_column = "Overcapacity"  # Update based on your dataset
columns_to_drop = ["Year", target_column]

# Separate features and target
X = data.drop(columns=[target_column])
y = data[target_column].reset_index(drop=True)

# Split training and testing data
train_data = data[data["Year"] < 2018]
test_data = data[data["Year"] >= 2018]

X_train = train_data.drop(columns=columns_to_drop)
y_train = train_data[target_column]

X_test = test_data.drop(columns=columns_to_drop)
y_test = test_data[target_column]

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the model
model = load_model("roanne_fnn_model.h5")

# Sidebar for user interaction
st.sidebar.title("Hospital Overcapacity Prediction")
selected_year = st.sidebar.number_input("Enter Year to Predict (e.g., 2025)", min_value=2006, max_value=2023, step=1)

if st.sidebar.button("Predict for Selected Year"):
    try:
        # Extract features for the selected year
        year_data = X[X["Year"] == selected_year]
        year_scaled = scaler.transform(year_data)
        predictions = model.predict(year_scaled)

        # Display results
        st.subheader(f"Prediction for {selected_year}")
        results = pd.DataFrame({
            "Actual": y[X["Year"] == selected_year],
            "Predicted": predictions.flatten()
        })
        st.write(results)
    except Exception as e:
        st.error(f"Error in processing input: {e}")

# Test set evaluation
if st.sidebar.button("Show Test Set Evaluation"):
    try:
        y_test_pred = model.predict(X_test_scaled)
        st.subheader("Test Set Evaluation")
        test_results = pd.DataFrame({
            "Year": test_data["Year"].values,
            "Actual": y_test.values,
            "Predicted": y_test_pred.flatten()
        })
        st.write(test_results)

        # Additional metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse = mean_squared_error(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"Mean Absolute Error: {mae:.4f}")
    except Exception as e:
        st.error(f"Error during test set evaluation: {e}")
