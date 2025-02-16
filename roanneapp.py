import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier  # Example model
import joblib  # For loading pre-trained models


def create_sample_data():
    """Create sample data for demonstration"""
    # Generate dates from 2019 to 2023
    dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='M')

    # Create base data
    np.random.seed(42)  # For reproducibility
    data = pd.DataFrame({
        'Date': dates,
        'Admissions': np.random.randint(400000, 700000, len(dates)),
        'Number of Beds': np.random.randint(1000, 20000, len(dates)),
        'Admissions per Bed': np.random.uniform(20, 50, len(dates)),
        'Year': dates.year
    })

    # Add overcapacity flag for demonstration
    data['Overcapacity'] = (data['Admissions per Bed'] > 30).astype(int)

    return data


def load_trained_model():
    """Load the trained model."""
    try:
        model = joblib.load("hospital_capacity_model.pkl")  # Replace with your model's path
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is in the correct location.")
        return None


def predict_overcapacity(model, input_data):
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
    st.title("Hospital Capacity Prediction")
    st.markdown("Analyzing historical hospital capacity trends and predicting overcapacity.")

    # Create sample data
    data = create_sample_data()

    # Filter options moved below the title
    st.subheader("Filters")
    selected_years = st.multiselect(
        "Select Years",
        options=sorted(data['Year'].unique()),
        default=sorted(data['Year'].unique())
    )

    # Filter data based on selection
    filtered_data = data[data['Year'].isin(selected_years)]

    # Prediction Section
    st.sidebar.subheader("Upload Testing Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file (2019-2023)", type=["csv"])
    if uploaded_file is not None:
        # Read uploaded file
        test_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Testing Data")
        st.write(test_data.head())

        # Ensure columns match expected input
        required_columns = ['Admissions', 'Number of Beds', 'Admissions per Bed']
        if all(col in test_data.columns for col in required_columns):
            # Load pre-trained model
            model = load_trained_model()
            if model is not None:
                # Make predictions
                test_data['Overcapacity_Prediction'] = predict_overcapacity(model, test_data)
                test_data['Overcapacity_Prediction'] = test_data['Overcapacity_Prediction'].apply(
                    lambda x: "Yes" if x == 1 else "No"
                )

                st.subheader("Predicted Results")
                st.write(test_data)

                # Visualize the results
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=test_data['Facility Type'],
                    y=test_data['Admissions'],
                    name="Admissions"
                ))
                fig.add_trace(go.Scatter(
                    x=test_data['Facility Type'],
                    y=[30] * len(test_data),  # Example threshold for overcapacity
                    mode='lines',
                    name="Overcapacity Threshold",
                    line=dict(color="red", dash="dash")
                ))

                fig.update_layout(
                    title="Predicted Overcapacity Results",
                    xaxis_title="Facility Type",
                    yaxis_title="Admissions per Bed",
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Uploaded file must contain the following columns: {', '.join(required_columns)}")

    # Create main visualizations
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Hospital Capacity Trends")

        fig = go.Figure()

        # Add admissions line
        fig.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['Admissions'],
            name='Admissions',
            line=dict(color='blue', width=2)
        ))

        # Add overcapacity threshold line
        fig.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=[30] * len(filtered_data),  # Example threshold
            name='Overcapacity Threshold',
            line=dict(color='red', width=1, dash='dot')
        ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Admissions per Bed",
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

        # Calculate metrics
        avg_admissions = filtered_data['Admissions'].mean()
        overcapacity_days = filtered_data['Overcapacity'].sum()

        # Display metrics
        st.metric("Average Admissions", f"{avg_admissions:.1f}")
        st.metric("Days Over Capacity", int(overcapacity_days))

    # Monthly Analysis
    st.subheader("Monthly Average Admissions per Bed")
    monthly_avg = filtered_data.groupby('Month')[['Admissions per Bed']].mean()
    monthly_avg = monthly_avg.reindex(pd.date_range(start='2023-01-01', periods=12, freq='M').strftime('%B'))

    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=monthly_avg.index,
        y=monthly_avg['Admissions per Bed'],
        name='Admissions per Bed',
        marker_color='blue'
    ))

    fig_monthly.update_layout(
        barmode='group',
        xaxis_title="Month",
        yaxis_title="Average Admissions per Bed"
    )

    st.plotly_chart(fig_monthly, use_container_width=True)


if __name__ == "__main__":
    main()
