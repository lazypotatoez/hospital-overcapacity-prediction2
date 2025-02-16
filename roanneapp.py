import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
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

# Define the expected columns and preprocessing pipeline
categorical_columns = ['Admission Type', 'Service Type', 'Facility Type']
numeric_columns = ['Admissions', 'Hospital Admissions', 'Number of Beds']

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns),  # One-hot encode categorical columns
        ('num', 'passthrough', numeric_columns)         # Keep numeric columns as-is
    ]
)

# Upload dataset
uploaded_file = st.file_uploader("Upload a Test Dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        test_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:")
        st.write(test_data.head())

        # Check for missing columns
        expected_columns = categorical_columns + numeric_columns
        missing_columns = [col for col in expected_columns if col not in test_data.columns]
        if missing_columns:
            st.error(f"The following required columns are missing: {missing_columns}")
            st.stop()

        # Apply preprocessing
        test_data_transformed = preprocessor.fit_transform(test_data)  # Transform test data

        # Debugging: Check the shape of the transformed data
        st.write("Shape of test_data_transformed:", test_data_transformed.shape)

        # Ensure correct format for LSTM input (3D)
        if len(test_data_transformed.shape) == 2:  # Ensure it is 2D
            X_test = np.expand_dims(test_data_transformed, axis=1)  # Convert to 3D for LSTM
        else:
            st.error("Preprocessed test data is not 2D. Please check your test file.")
            st.stop()

        # Predict with the model
        predictions = model.predict(X_test)
        predicted_labels = (predictions > 0.5).astype(int)

        # Display predictions
        st.subheader("Predictions:")
        st.write(predicted_labels)

        # Optionally check if 'Overcapacity' exists in the file for evaluation
        if 'Overcapacity' in test_data.columns:
            y_test = test_data['Overcapacity'].values  # Ground truth labels
            st.subheader("Performance Metrics:")
            st.text("Classification Report:")
            st.text(classification_report(y_test, predicted_labels.argmax(axis=1), zero_division=0))

            st.text("Confusion Matrix:")
            st.text(confusion_matrix(y_test, predicted_labels.argmax(axis=1)))

    except Exception as e:
        st.error(f"Error processing the test data: {e}")
