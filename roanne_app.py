import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load dataset
data = pd.read_csv("Dataset_with_Overcapacity.csv")

# Select numeric columns and target column
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
target_column = "Overcapacity"  # Replace with your actual target column name

# Ensure `X` and `y` are aligned
X = data[numeric_columns]
y = data[target_column].reset_index(drop=True)  # Reset index to align with `X`

# Add the `Year` column to `X` for filtering
X["Year"] = data["Year"].reset_index(drop=True)

# Split based on Year
train_indices = X["Year"] < 2018
test_indices = X["Year"] >= 2018

X_train = X[train_indices].drop(columns=["Year"])  # Training data before 2018
X_test = X[test_indices].drop(columns=["Year"])    # Testing data for 2018–2023
y_train = y[train_indices]
y_test = y[test_indices]

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the pre-trained model
model = load_model("roanne_fnn_model.h5")

# Streamlit app layout
st.title("Hospital Overcapacity Prediction")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a mode", ["Predict for Year", "Evaluate on Test Data"])

if app_mode == "Predict for Year":
    # Input for selecting year
    selected_year = st.number_input("Enter Year to Predict (e.g., 2025)", min_value=2006, max_value=2023, step=1)
    
    # Filter data for the selected year
    if selected_year in X["Year"].unique():
        year_data = X[X["Year"] == selected_year].drop(columns=["Year"])
        year_scaled = scaler.transform(year_data)
        predictions = model.predict(year_scaled)
        predicted_classes = (predictions > 0.5).astype(int)
        
        # Display predictions
        results = pd.DataFrame({
            "Actual": y[X["Year"] == selected_year].values,
            "Predicted": predicted_classes.flatten()
        })
        st.write(f"Predictions for the Year {selected_year}")
        st.dataframe(results)
    else:
        st.warning("No data available for the selected year.")

elif app_mode == "Evaluate on Test Data":
    # Predict on test data
    predictions = model.predict(X_test_scaled)
    predicted_classes = (predictions > 0.5).astype(int)
    
    # Show actual vs predicted
    results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predicted_classes.flatten()
    })
    st.write("Evaluation on Test Data (2018–2023)")
    st.dataframe(results)
    
    # Calculate evaluation metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_test, predicted_classes)
    precision = precision_score(y_test, predicted_classes)
    recall = recall_score(y_test, predicted_classes)
    f1 = f1_score(y_test, predicted_classes)
    
    # Display metrics
    st.write("Evaluation Metrics:")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
