import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the dataset and model
data = pd.read_csv("Dataset_with_Overcapacity.csv")
model = load_model("roanne_fnn_model.h5")

# Define columns to drop during scaling
columns_to_drop = ["Year", "Overcapacity"]

# Sidebar layout for navigation
st.sidebar.title("Hospital Overcapacity Prediction")
year = st.sidebar.slider("Enter Year to Predict (e.g., 2018 - 2023)", min_value=2018, max_value=2023, value=2023)

# Prepare features for train-test split
target_column = "Overcapacity"
X = data.drop(columns=[target_column])
y = data[target_column]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.drop(columns=columns_to_drop, axis=1))

# Split train-test data (testing for years 2018-2023)
train_indices = X["Year"] < 2018
test_indices = X["Year"] >= 2018

X_train = X_scaled[train_indices]
X_test = X_scaled[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

# Main page layout
st.title("Hospital Overcapacity Prediction")
st.write("Predicting overcapacity based on year and other features.")

if st.button("Predict for Selected Year"):
    # Prepare year input for prediction
    try:
        year_input = pd.DataFrame({"Year": [year]})
        year_scaled = scaler.transform(year_input)  # Transform using the same scaler
        predictions = model.predict(year_scaled)
        predicted_value = predictions[0][0]

        st.subheader(f"Prediction for Year {year}")
        st.write(f"Predicted Overcapacity: {'Yes' if predicted_value > 0.5 else 'No'}")
        st.write(f"Confidence Score: {predicted_value:.2f}")
    except ValueError as e:
        st.error(f"Error in processing input: {e}")

if st.button("Show Test Set Evaluation"):
    # Predict on the test set
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
