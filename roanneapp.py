import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.figure_factory as ff

def setup_page():
    st.set_page_config(page_title="Hospital Capacity Analysis", 
                      page_icon="🏥",
                      layout="wide")
    st.title("🏥 Hospital Capacity Analysis Dashboard")
    st.markdown("Advanced capacity analysis with predictive insights")

def create_sample_data():
    """Create enhanced sample data with more features"""
    dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='D')
    
    data = pd.DataFrame({
        'Date': dates,
        'Actual_Capacity': np.random.normal(75, 15, len(dates)),
        'Year': dates.year,
        'Month': dates.month_name(),
        'DayOfWeek': dates.day_name(),
        'IsWeekend': dates.weekday >= 5,
        'Season': pd.cut(dates.month, bins=[0,3,6,9,12], labels=['Winter','Spring','Summer','Fall'])
    })
    
    # Add seasonal patterns
    data['Actual_Capacity'] += np.sin(data.index * 2 * np.pi / 365) * 10
    
    # Add weekend effect
    data.loc[data['IsWeekend'], 'Actual_Capacity'] += 5
    
    # Add trend
    data['Actual_Capacity'] += data.index * 0.01
    
    # Add holiday effects (example dates)
    holidays = ['2019-12-25', '2020-12-25', '2021-12-25', '2022-12-25', '2023-12-25',
                '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01']
    data.loc[data['Date'].astype(str).isin(holidays), 'Actual_Capacity'] += 15
    
    # Generate predictions with different accuracy levels based on proximity to current date
    data['Predicted_Capacity'] = data['Actual_Capacity'] + np.random.normal(0, 
        np.linspace(3, 8, len(dates)))  # Increasing uncertainty over time
    
    # Add confidence intervals
    data['Prediction_Lower'] = data['Predicted_Capacity'] - np.linspace(2, 10, len(dates))
    data['Prediction_Upper'] = data['Predicted_Capacity'] + np.linspace(2, 10, len(dates))
    
    # Ensure values are between 0 and 100
    for col in ['Actual_Capacity', 'Predicted_Capacity', 'Prediction_Lower', 'Prediction_Upper']:
        data[col] = data[col].clip(0, 100)
    
    # Add risk levels
    data['Risk_Level'] = pd.cut(data['Actual_Capacity'], 
                               bins=[0, 60, 75, 85, 100],
                               labels=['Low', 'Moderate', 'High', 'Critical'])
    
    return data

def calculate_metrics(data):
    """Calculate advanced metrics for capacity analysis"""
    metrics = {
        'avg_capacity': data['Actual_Capacity'].mean(),
        'peak_capacity': data['Actual_Capacity'].max(),
        'std_dev': data['Actual_Capacity'].std(),
        'mae': mean_absolute_error(data['Actual_Capacity'], data['Predicted_Capacity']),
        'rmse': np.sqrt(mean_squared_error(data['Actual_Capacity'], data['Predicted_Capacity'])),
        'overcapacity_days': (data['Actual_Capacity'] > 80).sum(),
        'weekend_avg': data[data['IsWeekend']]['Actual_Capacity'].mean(),
        'weekday_avg': data[~data['IsWeekend']]['Actual_Capacity'].mean()
    }
    return metrics

def main():
    setup_page()
    
    # Create sample data
    data = create_sample_data()
    
    # Sidebar controls
    st.sidebar.header("Analysis Controls")
    
    # Date range selector
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(data['Date'].min(), data['Date'].max()),
        min_value=data['Date'].min(),
        max_value=data['Date'].max()
    )
    
    # Additional filters
    view_mode = st.sidebar.selectbox(
        "View Mode",
        ["Daily", "Weekly", "Monthly"]
    )
    
    risk_levels = st.sidebar.multiselect(
        "Risk Levels",
        options=['Low', 'Moderate', 'High', 'Critical'],
        default=['High', 'Critical']
    )
    
    show_confidence = st.sidebar.checkbox("Show Prediction Confidence Intervals", value=True)
    
    # Filter data
    mask = (data['Date'] >= pd.Timestamp(date_range[0])) & \
           (data['Date'] <= pd.Timestamp(date_range[1])) & \
           (data['Risk_Level'].isin(risk_levels))
    filtered_data = data[mask]
    
    # Resample data based on view mode
    if view_mode == "Weekly":
        filtered_data = filtered_data.resample('W', on='Date').mean().reset_index()
    elif view_mode == "Monthly":
        filtered_data = filtered_data.resample('M', on='Date').mean().reset_index()
    
    # Main dashboard
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Capacity Forecast Analysis")
        
        fig = go.Figure()
        
        # Add actual capacity
        fig.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['Actual_Capacity'],
            name='Actual Capacity',
            line=dict(color='blue', width=2)
        ))
        
        # Add predicted capacity with confidence interval
        fig.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['Predicted_Capacity'],
            name='Predicted Capacity',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        if show_confidence:
            fig.add_trace(go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['Prediction_Upper'],
                name='Upper Bound',
                line=dict(color='rgba(255,0,0,0.2)', width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['Prediction_Lower'],
                name='Lower Bound',
                line=dict(color='rgba(255,0,0,0.2)', width=0),
                fill='tonexty',
                showlegend=False
            ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Capacity (%)",
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Performance Metrics")
        metrics = calculate_metrics(filtered_data)
        
        cols = st.columns(2)
        cols[0].metric("Average Capacity", f"{metrics['avg_capacity']:.1f}%")
        cols[1].metric("Peak Capacity", f"{metrics['peak_capacity']:.1f}%")
        cols[0].metric("Weekend Average", f"{metrics['weekend_avg']:.1f}%")
        cols[1].metric("Weekday Average", f"{metrics['weekday_avg']:.1f}%")
        cols[0].metric("Prediction MAE", f"{metrics['mae']:.2f}")
        cols[1].metric("Prediction RMSE", f"{metrics['rmse']:.2f}")
        
        # Add forecast reliability score
        reliability_score = 100 - (metrics['rmse'] * 2)  # Simple example calculation
        st.progress(reliability_score/100)
        st.caption(f"Forecast Reliability Score: {reliability_score:.1f}%")
    
    # Additional analysis sections
    st.subheader("Capacity Distribution Analysis")
    
    # Create distribution plot
    fig_dist = ff.create_distplot(
        [filtered_data['Actual_Capacity'], filtered_data['Predicted_Capacity']], 
        ['Actual', 'Predicted'],
        bin_size=2
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Seasonal patterns
    st.subheader("Seasonal Pattern Analysis")
    seasonal_fig = px.box(filtered_data, x='Season', y='Actual_Capacity', color='Season')
    st.plotly_chart(seasonal_fig, use_container_width=True)

if __name__ == "__main__":