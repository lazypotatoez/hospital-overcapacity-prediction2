import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Load the trained models
fnn_model = load_model("roanne_fnn_model.h5")  # Update to your FNN model filename
scaler = StandardScaler()

# Load and scale data
data = pd.read_csv("Final_Merged_Dataset.csv")
scaler.fit(data.drop(["Year", "Overcapacity"], axis=1))  # Fit scaler without non-numeric columns

# Sidebar layout for navigation
st.sidebar.title("Hospital Overcapacity Prediction")
st.sidebar.markdown("Navigate through the app:")
navigation = st.sidebar.radio("Sections", ["Single Prediction", "Batch Prediction", "Visualizations", "Model Performance", "How It Works"])

# Function to preprocess input
def preprocess_input(user_input):
    try:
        features = np.array(user_input.split(",")).astype(float).reshape(1, -1)
        if features.shape[1] != 21:  # Adjust this if feature count changes
            return None, "Please provide exactly 21 features."
        scaled_features = scaler.transform(features)
        return scaled_features, None
    except ValueError:
        return None, "Invalid input. Please ensure you provide numeric values."

# Function to predict
def predict_overcapacity(model, features):
    prediction_prob = model.predict(features)[0][0]
    prediction = "Overcapacity" if prediction_prob > 0.5 else "Not Overcapacity"
    confidence = prediction_prob if prediction == "Overcapacity" else 1 - prediction_prob
    return prediction, confidence

# Main sections
if navigation == "Single Prediction":
    st.title("Hospital Overcapacity Prediction")
    st.write("Enter feature values (comma-separated):")
    user_input = st.text_input("Feature values", placeholder="e.g., 100,200,300,...")

    if st.button("Predict"):
        if not user_input:
            st.error("Please provide feature values.")
        else:
            features, error = preprocess_input(user_input)
            if error:
                st.error(error)
            else:
                prediction, confidence = predict_overcapacity(fnn_model, features)
                st.write(f"**Prediction:** {prediction}")
                st.write(f"**Confidence:** {confidence:.2f}")

elif navigation == "Batch Prediction":
    st.title("Batch Predictions")
    uploaded_file = st.file_uploader("Upload a CSV for Batch Prediction", type="csv")
    
    if uploaded_file:
        batch_data = pd.read_csv(uploaded_file)
        try:
            scaled_batch = scaler.transform(batch_data)
            batch_predictions = fnn_model.predict(scaled_batch)
            batch_data["Prediction"] = ["Overcapacity" if p > 0.5 else "Not Overcapacity" for p in batch_predictions]
            st.dataframe(batch_data)
            st.download_button("Download Predictions", batch_data.to_csv(index=False), "predictions.csv")
        except Exception as e:
            st.error(f"Error in processing: {str(e)}")

elif navigation == "Visualizations":
    st.title("Data Visualizations")
    st.write("### Dataset Overview")
    st.dataframe(data)
    
    st.write("### Feature Correlation Heatmap")
    corr = data.drop(["Year", "Overcapacity"], axis=1).corr()
    fig = px.imshow(corr, title="Feature Correlation Heatmap")
    st.plotly_chart(fig)

    st.write("### Admissions vs. Beds Over Time")
    fig = px.line(data, x="Year", y=["Admissions", "Number of Beds"], title="Admissions vs. Beds Over Time")
    st.plotly_chart(fig)

elif navigation == "Model Performance":
    st.title("Model Performance")
    st.write("### FNN Model")
    st.write(f"**Test Accuracy:** 0.8365")  # Replace with actual metrics
    st.write(f"**Test Loss:** 0.2401")  # Replace with actual metrics
    
    # Placeholder for performance graphs
    st.write("### Training and Validation Accuracy")
    fig = px.line(
        {"Train Accuracy": [0.75, 0.80, 0.83], "Val Accuracy": [0.73, 0.79, 0.83]},
        labels={"value": "Accuracy", "index": "Epoch"},
        title="Training and Validation Accuracy"
    )
    st.plotly_chart(fig)

elif navigation == "How It Works":
    st.title("How It Works")
    st.write("""
    ### About This App
    This app predicts hospital overcapacity based on 21 input features. 
    The predictions are made using a Feedforward Neural Network (FNN) model.

    ### Steps:
    1. Provide input features for single predictions or upload a CSV for batch predictions.
    2. Models return predictions with confidence scores.
    3. Visualize data or compare model performance.
    """)
    st.write("""
    ### Features:
    - Single and batch prediction support.
    - Interactive data visualizations.
    - Downloadable prediction results.
    """)

