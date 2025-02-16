import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf

# Set page config
st.set_page_config(
    page_title="Hospital Overcapacity Prediction",
    page_icon="🏥",
    layout="wide"
)

# Function to preprocess data
def preprocess_data(df):
    """Preprocess the input data to match training pipeline"""
    # Create a copy of the dataframe
    processed_df = df.copy()
    
    # Convert 'Admissions per Bed' to numeric, removing any text after the number
    processed_df['Admissions per Bed'] = pd.to_numeric(
        processed_df['Admissions per Bed'].astype(str).str.extract('(\d+\.?\d*)')[0]
    )
    
    # Label encode categorical columns
    categorical_columns = ['Admission Type', 'Service Type', 'Facility Type']
    label_encoders = {}
    
    for column in categorical_columns:
        if column in processed_df.columns:
            label_encoders[column] = LabelEncoder()
            processed_df[column] = label_encoders[column].fit_transform(processed_df[column])
    
    # Convert numeric columns to float
    numeric_columns = ['Admissions', 'Hospital Admissions', 'Number of Beds', 'Admissions per Bed']
    for column in numeric_columns:
        if column in processed_df.columns:
            processed_df[column] = pd.to_numeric(processed_df[column], errors='coerce')
    
    # Scale numeric features
    scaler = StandardScaler()
    processed_df[numeric_columns] = scaler.fit_transform(processed_df[numeric_columns])
    
    # Reshape data for LSTM (samples, timesteps, features)
    n_features = processed_df.shape[1]
    n_timesteps = 1  # Adjust based on your model's input shape
    reshaped_data = processed_df.values.reshape(-1, n_timesteps, n_features)
    
    return reshaped_data

def predict_overcapacity(model, features):
    """Make predictions using the loaded model"""
    predictions = model.predict(features)
    # Get the predicted class (0 or 1)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes

def main():
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
            # Load data
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
            results_df['Predicted_Status'] = results_df['Predicted_Overcapacity'].map({
                0: 'No Overcapacity',
                1: 'Overcapacity'
            })

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
            st.write("Detailed error:", str(e))
            st.write("Please ensure your input data matches the expected format.")

    else:
        st.info("Please upload your test data file to get predictions.")

if __name__ == "__main__":
    main()