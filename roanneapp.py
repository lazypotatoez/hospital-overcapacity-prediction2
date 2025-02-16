import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def setup_page():
    st.set_page_config(page_title="Hospital Capacity Analysis", 
                      page_icon="🏥",
                      layout="wide")
    st.title("🏥 Hospital Capacity Analysis (2019-2023)")
    st.markdown("Analyzing historical hospital capacity trends and predictions")

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

def main():
    setup_page()
    
    # Create sample data
    data = create_sample_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_years = st.sidebar.multiselect(
        "Select Years",
        options=sorted(data['Year'].unique()),
        default=sorted(data['Year'].unique())
    )
    
    # Filter data based on selection
    filtered_data = data[data['Year'].isin(selected_years)]
    
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
        accuracy = (filtered_data['Actual_Overcapacity'] == 
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