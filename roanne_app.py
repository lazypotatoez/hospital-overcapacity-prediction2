# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import streamlit as st

# Load the dataset
data = pd.read_csv("Final_Merged_Dataset.csv")

# Define numeric columns (exclude 'Year' or any non-numeric columns if necessary)
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
columns_to_drop = ["Year"]  # Add other non-feature columns if applicable
if "Overcapacity" in data.columns:
    target_column = "Overcapacity"
    numeric_columns.remove(target_column)
else:
    target_column = None

# Split data into train and test sets
X = data[numeric_columns]
y = data[target_column] if target_column else None
X["Year"] = data["Year"]  # Keep the year column for filtering
X_train = X[X["Year"] < 2018].drop(columns=["Year"])  # Training data before 2018
X_test = X[X["Year"] >= 2018].drop(columns=["Year"])  # Testing data for 2018–2023
y_train = y[X["Year"] < 2018]
y_test = y[X["Year"] >= 2018]

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the trained model
model = load_model("roanne_fnn_model.h5")

# Sidebar for selecting the year
st.sidebar.title("Year Selection")
selected_year = st.sidebar.slider(
    "Select a year to view predictions (2018–2023):", min_value=2018, max_value=2023, value=2023
)

# Filter test data based on the selected year
filtered_X_test = X_test[X["Year"] == selected_year]
filtered_y_test = y_test[X["Year"] == selected_year]

# Display predictions and actuals
if not filtered_X_test.empty:
    st.title(f"Predictions for the Year {selected_year}")
    predictions = model.predict(scaler.transform(filtered_X_test)).flatten()
    predicted_classes = (predictions > 0.5).astype(int)

    # Combine predictions and actuals for display
    results_df = pd.DataFrame({
        "Actual": filtered_y_test.values,
        "Predicted": predicted_classes,
        "Prediction Score": predictions
    })

    st.write(f"Testing Results for Year {selected_year}")
    st.dataframe(results_df)

    # Display metrics for the selected year
    accuracy = np.mean(predicted_classes == filtered_y_test.values)
    st.write(f"Accuracy for {selected_year}: {accuracy:.2%}")
else:
    st.write(f"No test data available for the year {selected_year}.")
