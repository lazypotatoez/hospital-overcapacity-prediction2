import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the dataset and model
data = pd.read_csv("Dataset_with_Overcapacity.csv")
model = load_model("roanne_fnn_model.h5")

# Debug: Show column names in dataset
st.write("Columns in dataset:", data.columns.tolist())

# Define target and features
target_column = "Overcapacity"
columns_to_drop = ["Year"]  # Ensure "Overcapacity" is not dropped here

# Drop rows with missing values and ensure all columns are numeric before scaling
data = data.dropna()
numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()

# Ensure that only numeric columns are passed to the scaler
try:
    X = data.drop(columns=[target_column])  # Drop target column for features
    y = data[target_column]

    X_scaled = StandardScaler().fit_transform(X[numeric_columns])
except KeyError as e:
    st.error(f"Key error: {e}. Ensure target and numeric columns are correct.")
    st.stop()
except ValueError as e:
    st.error(f"Value error during scaling: {e}. Check for non-numeric or missing data.")
    st.stop()

# Sidebar layout for navigation
st.sidebar.title("Hospital Overcapacity Prediction")
year = st.sidebar.slider("Enter Year to Predict (e.g., 2018 - 2023)", min_value=2018, max_value=2023, value=2023)

# Split train-test data
try:
    train_indices = data["Year"] < 2018
    test_indices = data["Year"] >= 2018

    X_train = X_scaled[train_indices]
    X_test = X_scaled[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
except KeyError as e:
    st.error(f"Error in data splitting: {e}")
    st.stop()

# Main page layout
st.title("Hospital Overcapacity Prediction")
st.write("Predicting overcapacity based on year and other features.")

if st.button("Predict for Selected Year"):
    # Prepare year input for prediction
    try:
        year_input = pd.DataFrame({"Year": [year]})
        year_scaled = StandardScaler().fit_transform(year_input)  # Transform using the same scaler
        predictions = model.predict(year_scaled)
        predicted_value = predictions[0][0]

        st.subheader(f"Prediction for Year {year}")
        st.write(f"Predicted Overcapacity: {'Yes' if predicted_value > 0.5 else 'No'}")
        st.write(f"Confidence Score: {predicted_value:.2f}")
    except ValueError as e:
        st.error(f"Error in processing input: {e}")

if st.button("Show Test Set Evaluation"):
    # Predict on the test set
    try:
        y_test_predictions = (model.predict(X_test) > 0.5).astype(int)

        # Display actual vs predicted
        results = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": y_test_predictions.flatten()
        })
        st.write("Test Set Evaluation")
        st.dataframe(results)

        # Calculate and display metrics
        accuracy = (results["Actual"] == results["Predicted"]).mean()
        st.write(f"Test Set Accuracy: {accuracy:.2%}")
    except Exception as e:
        st.error(f"Error during test set evaluation: {e}")
