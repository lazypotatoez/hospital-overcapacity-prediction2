import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Title of the app
st.title("Overcapacity and Admissions Prediction App")

# Load the saved model
model = load_model('lstm_model.h5')
st.write("Model loaded successfully.")

# Upload dataset
uploaded_file = st.file_uploader("Upload a Test Dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    # Read the uploaded CSV file
    test_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset:")
    st.write(test_data.head())

    # Prepare the input for the model
    if 'Overcapacity' in test_data.columns:
        test_data = test_data.drop(columns=['Overcapacity'])  # Drop target column if it exists

    # Ensure test data is in the correct format (e.g., 3D for LSTM)
    X_test = np.expand_dims(test_data.values, axis=1)

    # Predict with the model
    predictions = model.predict(X_test)
    predicted_labels = (predictions > 0.5).astype(int)

    # Display predictions
    st.subheader("Predictions:")
    st.write(predicted_labels)

    # Add performance metrics if ground truth labels are available
    if 'Overcapacity' in uploaded_file:
        y_test = uploaded_file['Overcapacity']
        st.subheader("Performance Metrics:")
        st.text("Classification Report:")
        st.text(classification_report(y_test, predicted_labels))

        st.text("Confusion Matrix:")
        st.text(confusion_matrix(y_test, predicted_labels))
