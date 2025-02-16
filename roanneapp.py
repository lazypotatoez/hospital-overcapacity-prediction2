import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Set page config
st.set_page_config(
    page_title="Hospital Overcapacity Prediction",
    page_icon="🏥",
    layout="wide"
)

# Function to load and preprocess data
def preprocess_data(df):
    """Preprocess the input data similar to training pipeline"""
    # Add your preprocessing steps here based on your training pipeline
    # For example:
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

# Function to make predictions
def predict_overcapacity(model, features):
    """Make predictions using the loaded model"""
    predictions = model.predict(features)
    # Convert predictions to binary (0 or 1) based on threshold
    binary_predictions = (predictions > 0.5).astype(int)
    return binary_predictions

def main():
    # Title and description
    st.title("🏥 Hospital Overcapacity Prediction System")
    st.markdown("""
    This application predicts hospital overcapacity based on historical data and current metrics.
    Upload your test data to get predictions.
    """)

    # Sidebar
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose your test features CSV file", type="csv")

    try:
        # Load the model
        model = load_model('roanne_lstm_model.h5')
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return

    if uploaded_file is not None:
        try:
            # Load and preprocess data
            test_data = pd.read_csv(uploaded_file)
            st.subheader("Input Data Preview")
            st.dataframe(test_data.head())

            # Preprocess the data
            processed_data = preprocess_data(test_data)

            # Make predictions
            predictions = predict_overcapacity(model, processed_data)

            # Display results
            st.subheader("Predictions")
            results_df = test_data.copy()
            results_df['Predicted_Overcapacity'] = predictions
            results_df['Predicted_Status'] = results_df['Predicted_Overcapacity'].map({0: 'No Overcapacity', 1: 'Overcapacity'})

            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Predictions", len(predictions))
                st.metric("Predicted Overcapacity Cases", np.sum(predictions))
            with col2:
                st.metric("Normal Capacity Cases", len(predictions) - np.sum(predictions))
                st.metric("Overcapacity Percentage", f"{(np.sum(predictions)/len(predictions)*100):.2f}%")

            # Show detailed results
            st.dataframe(results_df)

            # Download predictions
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="hospital_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.write("Please ensure your input data matches the expected format.")

    else:
        st.info("Please upload your test data file to get predictions.")

if __name__ == "__main__":
    main()