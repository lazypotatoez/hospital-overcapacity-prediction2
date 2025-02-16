import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor  # Example model for demonstration
import joblib  # For loading pre-trained models


def create_sample_data():
    """Create sample data for demonstration"""
    # Generate dates from 2019 to 2023
    dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='M')

    # Create base data
    np.random.seed(42)  # For reproducibility
    data = pd.DataFrame({
        'Date': dates,
        'Actual_Capacity': np.random.normal(75, 15, len(dates)),  # Mean of 75% with some variation
        'Year': dates.year,
        'Month': dates.month_name()
    })

    # Add some seasonal patterns
    data['Actual_Capacity'] += np.sin(data.index * 2 * np.pi / 12) * 5

    # Add trend (slight increase over time)
    data['Actual_Capacity'] += data.index * 0.03

    # Add predicted values with some deviation from actual
    data['Predicted_Capacity'] = data['Actual_Capacity'] + np.random.normal(0, 5, len(dates))

    # Ensure values are between 0 and 100
    data['Actual_Capacity'] = data['Actual_Capacity'].clip(0, 100)
    data['Predicted_Capacity'] = data['Predicted_Capacity'].clip(0, 100)

    # Add overcapacity flags
    threshold = 80
    data['Actual_Overcapacity'] = data['Actual_Capacity'] > threshold
    data['Predicted_Overcapacity'] = data['Predicted_Capacity'] > threshold

    return data


def load_model():
    """Load the pre-trained model"""
    # Replace with your model path
    model_path = "hospital_capacity_model.pkl"
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def predict_capacity(model, input_data):
    """Make predictions using the model"""
    try:
        # Select relevant features
        features = input_data[['Admissions', 'Number of Beds', 'Admissions per Bed']]
        predictions = model.predict(features)
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None


def main():
    st.title("🏥 Hospital Capacity Prediction")
    st.markdown("Analyzing hospital capacity trends and making predictions")

    # Section: Upload Testing Data
    st.sidebar.subheader("Upload Testing Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Testing Data")
        st.write(test_data.head())

        # Ensure required columns are present in the uploaded file
        required_columns = ['Admissions', 'Number of Beds', 'Admissions per Bed']
        if all(col in test_data.columns for col in required_columns):
            try:
                # Load pre-trained model
                model = load_model()

                # Make predictions
                predictions = predict_capacity(model, test_data)

                if predictions is not None:
                    # Add predictions to the DataFrame
                    test_data['Predicted_Capacity'] = predictions

                    st.subheader("Predicted Results")
                    st.write(test_data)

                    # Visualize predictions
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=test_data['Facility Type'],  # Replace with appropriate column
                        y=test_data['Predicted_Capacity'],
                        name='Predicted Capacity',
                        marker_color='red'
                    ))

                    fig.update_layout(
                        xaxis_title="Facility Type",
                        yaxis_title="Predicted Capacity (%)",
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("No predictions were generated. Check the model and input data.")
            except Exception as e:
                st.error(f"Error loading model or making predictions: {e}")
        else:
            st.error(f"Uploaded file must contain the following columns: {', '.join(required_columns)}")
    else:
        st.info("Upload a CSV file to start predictions.")

    # Sample data visualization for reference
    data = create_sample_data()
    st.subheader("Sample Data Trends")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Hospital Capacity Trends")

        fig = go.Figure()

        # Add actual capacity line
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Actual_Capacity'],
            name='Actual Capacity',
            line=dict(color='blue', width=2)
        ))

        # Add predicted capacity line
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Predicted_Capacity'],
            name='Predicted Capacity',
            line=dict(color='red', width=2, dash='dash')
        ))

        # Add overcapacity threshold line
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=[80] * len(data),
            name='Overcapacity Threshold',
            line=dict(color='yellow', width=1, dash='dot')
        ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Capacity (%)",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Key Metrics")

        avg_actual = data['Actual_Capacity'].mean()
        avg_predicted = data['Predicted_Capacity'].mean()
        overcapacity_days_actual = data['Actual_Overcapacity'].sum()
        overcapacity_days_predicted = data['Predicted_Overcapacity'].sum()

        st.metric("Average Actual Capacity", f"{avg_actual:.1f}%")
        st.metric("Average Predicted Capacity", f"{avg_predicted:.1f}%")
        st.metric("Days Over Capacity (Actual)", int(overcapacity_days_actual))
        st.metric("Days Over Capacity (Predicted)", int(overcapacity_days_predicted))

if __name__ == "__main__":
    main()
