from sklearn.ensemble import RandomForestClassifier  # Example model
import joblib  # For loading the pre-trained model

# Add this function to load your pre-trained model
def load_trained_model():
    """Load the trained model."""
    try:
        model = joblib.load("hospital_capacity_model.pkl")  # Replace with your model's path
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is in the correct location.")
        return None

# Modify your main function to include testing CSV upload and prediction
def main():
    st.title("🏥 Hospital Capacity Prediction")
    st.markdown("Analyzing historical hospital capacity trends and predictions")

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
        required_columns = ['Date', 'Actual_Capacity']  # Adjust to your model's input requirements
        if all(col in test_data.columns for col in required_columns):
            # Load pre-trained model
            model = load_trained_model()
            if model is not None:
                # Prepare data for prediction
                features = test_data[['Actual_Capacity']]  # Adjust columns as per your model
                test_data['Overcapacity_Prediction'] = model.predict(features)
                test_data['Overcapacity_Prediction'] = test_data['Overcapacity_Prediction'].apply(
                    lambda x: "Yes" if x == 1 else "No"
                )

                st.subheader("Predicted Results")
                st.write(test_data)

                # Visualize the results
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=test_data['Date'],
                    y=test_data['Actual_Capacity'],
                    name="Actual Capacity"
                ))
                fig.add_trace(go.Scatter(
                    x=test_data['Date'],
                    y=[80] * len(test_data),
                    mode='lines',
                    name="Overcapacity Threshold",
                    line=dict(color="red", dash="dash")
                ))

                fig.update_layout(
                    title="Predicted Overcapacity Results",
                    xaxis_title="Date",
                    yaxis_title="Capacity (%)",
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

        # Add actual capacity line
        fig.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['Actual_Capacity'],
            name='Actual Capacity',
            line=dict(color='blue', width=2)
        ))

        # Add predicted capacity line
        fig.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['Predicted_Capacity'],
            name='Predicted Capacity',
            line=dict(color='red', width=2, dash='dash')
        ))

        # Add overcapacity threshold line
        fig.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=[80] * len(filtered_data),
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

        # Calculate metrics
        avg_actual = filtered_data['Actual_Capacity'].mean()
        avg_predicted = filtered_data['Predicted_Capacity'].mean()
        overcapacity_days_actual = filtered_data['Actual_Overcapacity'].sum()
        overcapacity_days_predicted = filtered_data['Predicted_Overcapacity'].sum()

        # Display metrics
        st.metric("Average Actual Capacity", f"{avg_actual:.1f}%")
        st.metric("Average Predicted Capacity", f"{avg_predicted:.1f}%")
        st.metric("Days Over Capacity (Actual)", int(overcapacity_days_actual))
        st.metric("Days Over Capacity (Predicted)", int(overcapacity_days_predicted))

        # Prediction Accuracy
        accuracy = (filtered_data['Actual_Overcapacity'] == \
                   filtered_data['Predicted_Overcapacity']).mean() * 100
        st.metric("Model Accuracy", f"{accuracy:.1f}%")

    # Monthly Analysis
    st.subheader("Monthly Average Capacity")
    monthly_avg = filtered_data.groupby('Month')[['Actual_Capacity', 'Predicted_Capacity']].mean()
    monthly_avg = monthly_avg.reindex(pd.date_range(start='2023-01-01', periods=12, freq='M').strftime('%B'))

    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=monthly_avg.index,
        y=monthly_avg['Actual_Capacity'],
        name='Actual',
        marker_color='blue'
    ))
    fig_monthly.add_trace(go.Bar(
        x=monthly_avg.index,
        y=monthly_avg['Predicted_Capacity'],
        name='Predicted',
        marker_color='red'
    ))

    fig_monthly.update_layout(
        barmode='group',
        xaxis_title="Month",
        yaxis_title="Average Capacity (%)"
    )

    st.plotly_chart(fig_monthly, use_container_width=True)


if __name__ == "__main__":
    main()
