import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Title of the app
st.title("Overcapacity and Admissions Prediction App")

# Load the saved model
try:
    model = load_model('./roanne_lstm_model.h5')  # Adjust path if needed
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Upload dataset
uploaded_file = st.file_uploader("Upload a Test Dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        test_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:")
        st.write(test_data.head())

        # Ensure numeric data
        test_data = test_data.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
        test_data = test_data.fillna(0)  # Handle missing values

        # Define expected columns based on training
        expected_columns = ['Admissions', 'Hospital Admissions', 'Number of Beds', 
                            'Admission Type', 'Service Type', 'Facility Type']

        # Ensure test data has only the expected features
        missing_columns = [col for col in expected_columns if col not in test_data.columns]
        if missing_columns:
            st.error(f"The following required columns are missing in the uploaded file: {missing_columns}")
            st.stop()

        test_data = test_data[expected_columns]

        # Prepare the input for the LSTM model (3D format)
        X_test = np.expand_dims(test_data.values, axis=1)

        # Predict with the model
        predictions = model.predict(X_test)
        predicted_labels = (predictions > 0.5).astype(int)

        # Display predictions
        st.subheader("Predictions:")
        st.write(predicted_labels)

        # Optionally check if 'Overcapacity' is included in the file for evaluation
        if 'Overcapacity' in test_data.columns:
            y_test = test_data['Overcapacity'].values  # Ground truth labels
            st.subheader("Performance Metrics:")
            st.text("Classification Report:")
            st.text(classification_report(y_test, predicted_labels.argmax(axis=1), zero_division=0))

            st.text("Confusion Matrix:")
            st.text(confusion_matrix(y_test, predicted_labels.argmax(axis=1)))

    except Exception as e:
        st.error(f"Error processing the test data: {e}")
