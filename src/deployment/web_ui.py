"""
Web UI for the Predictive Maintenance API.
This module provides a Streamlit-based web interface for interacting with the API.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config_loader import load_config

# Load configuration
config = load_config()

# API URL
API_URL = f"http://localhost:{config['deployment'].get('api_port', 8000)}"

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        background-color: #f8f9fa;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
    }
    .failure-high {
        color: #e74c3c;
    }
    .failure-medium {
        color: #f39c12;
    }
    .failure-low {
        color: #2ecc71;
    }
    .recommendation {
        background-color: #f8f9fa;
        border-left: 5px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """
    Check if the API is healthy.
    
    Returns:
        bool: True if API is healthy, False otherwise.
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_monitoring_stats():
    """
    Get monitoring statistics from the API.
    
    Returns:
        dict: Monitoring statistics.
    """
    try:
        response = requests.get(f"{API_URL}/monitoring/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def make_prediction(machine_data):
    """
    Make a prediction using the API.
    
    Args:
        machine_data: Machine data for prediction.
        
    Returns:
        dict: Prediction result.
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=machine_data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def make_batch_prediction(machines_data):
    """
    Make batch predictions using the API.
    
    Args:
        machines_data: List of machine data for batch prediction.
        
    Returns:
        dict: Batch prediction results.
    """
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"machines": machines_data},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error making batch prediction: {str(e)}")
        return None

def format_probability(prob):
    """
    Format probability as percentage with color coding.
    
    Args:
        prob: Probability value.
        
    Returns:
        str: HTML formatted probability.
    """
    if prob >= 0.7:
        return f'<span class="failure-high">{prob:.1%}</span>'
    elif prob >= 0.3:
        return f'<span class="failure-medium">{prob:.1%}</span>'
    else:
        return f'<span class="failure-low">{prob:.1%}</span>'

def display_dashboard():
    """
    Display the monitoring dashboard.
    """
    st.markdown('<h1 class="main-header">Predictive Maintenance Monitoring</h1>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ API is not available. Please make sure the API server is running.")
        if st.button("Retry Connection"):
            st.experimental_rerun()
        return
    
    # Get monitoring stats
    stats = get_monitoring_stats()
    
    if not stats or "message" in stats:
        st.info("No prediction data available yet. Make some predictions to see statistics.")
    else:
        # Display metrics in a row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{stats["total_predictions"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Total Predictions</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{stats["failure_predictions"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Detected Failures</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            failure_rate_color = "failure-high" if stats["failure_rate"] > 0.3 else "failure-low"
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value {failure_rate_color}">{stats["failure_rate"]:.1%}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Failure Rate</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{stats["recent_failure_rate"]:.1%}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Recent Failure Rate</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Failure type distribution
        if "failure_type_distribution" in stats and stats["failure_type_distribution"]:
            st.markdown('<h2 class="sub-header">Failure Type Distribution</h2>', unsafe_allow_html=True)
            
            failure_types = list(stats["failure_type_distribution"].keys())
            failure_counts = list(stats["failure_type_distribution"].values())
            
            fig = px.pie(
                values=failure_counts,
                names=failure_types,
                title="Failure Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4
            )
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=60, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)

def display_single_prediction():
    """
    Display the single prediction form.
    """
    st.markdown('<h1 class="main-header">Machine Failure Prediction</h1>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ API is not available. Please make sure the API server is running.")
        if st.button("Retry Connection"):
            st.experimental_rerun()
        return
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Enter Machine Parameters</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"], index=1)
        air_temp = st.slider("Air Temperature [K]", min_value=295.0, max_value=305.0, value=300.0, step=0.1)
        process_temp = st.slider("Process Temperature [K]", min_value=305.0, max_value=315.0, value=310.5, step=0.1)
    
    with col2:
        rotational_speed = st.slider("Rotational Speed [rpm]", min_value=1000, max_value=2000, value=1500, step=10)
        torque = st.slider("Torque [Nm]", min_value=30.0, max_value=60.0, value=40.0, step=0.5)
        tool_wear = st.slider("Tool Wear [min]", min_value=0, max_value=250, value=120, step=5)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Predict Failure"):
        with st.spinner("Making prediction..."):
            # Prepare machine data
            machine_data = {
                "machine_type": machine_type,
                "air_temperature": air_temp,
                "process_temperature": process_temp,
                "rotational_speed": rotational_speed,
                "torque": torque,
                "tool_wear": tool_wear
            }
            
            # Make prediction
            prediction = make_prediction(machine_data)
            
            if prediction:
                # Display prediction result
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h2 class="sub-header">Prediction Result</h2>', unsafe_allow_html=True)
                
                # Failure status
                if prediction["machine_failure"]:
                    st.error("⚠️ **FAILURE PREDICTED**")
                else:
                    st.success("✅ **NO FAILURE PREDICTED**")
                
                # Failure probability
                st.markdown(f"**Failure Probability:** {format_probability(prediction['failure_probability'])}", unsafe_allow_html=True)
                
                # Gauge chart for failure probability
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction["failure_probability"] * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Failure Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 30], "color": "green"},
                            {"range": [30, 70], "color": "orange"},
                            {"range": [70, 100], "color": "red"}
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": prediction["failure_probability"] * 100
                        }
                    }
                ))
                fig.update_layout(height=250, margin=dict(t=50, b=0, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Failure types
                if prediction["failure_type"]:
                    st.markdown('<h3>Failure Type Probabilities</h3>', unsafe_allow_html=True)
                    
                    failure_types = list(prediction["failure_type"].keys())
                    failure_probas = list(prediction["failure_type"].values())
                    
                    # Sort by probability
                    sorted_indices = np.argsort(failure_probas)[::-1]
                    failure_types = [failure_types[i] for i in sorted_indices]
                    failure_probas = [failure_probas[i] for i in sorted_indices]
                    
                    # Create a horizontal bar chart
                    fig = go.Figure()
                    
                    # Add bars
                    fig.add_trace(go.Bar(
                        x=failure_probas,
                        y=failure_types,
                        orientation='h',
                        marker=dict(
                            color=['#e74c3c' if p > 0.3 else '#f39c12' if p > 0.1 else '#2ecc71' for p in failure_probas],
                            line=dict(color='rgba(0, 0, 0, 0)', width=1)
                        )
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Failure Type Probabilities",
                        xaxis_title="Probability",
                        yaxis=dict(
                            title="Failure Type",
                            categoryorder='total ascending'
                        ),
                        height=300,
                        margin=dict(t=50, b=50, l=20, r=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Maintenance recommendation
                if prediction["maintenance_recommendation"]:
                    st.markdown('<div class="recommendation">', unsafe_allow_html=True)
                    st.markdown('<h3>Maintenance Recommendation</h3>', unsafe_allow_html=True)
                    st.markdown(f"<p>{prediction['maintenance_recommendation']}</p>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

def display_batch_prediction():
    """
    Display the batch prediction form.
    """
    st.markdown('<h1 class="main-header">Batch Prediction</h1>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ API is not available. Please make sure the API server is running.")
        if st.button("Retry Connection"):
            st.experimental_rerun()
        return
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Upload Machine Data CSV</h2>', unsafe_allow_html=True)
    
    st.info("""
    Upload a CSV file with machine data. The CSV should have the following columns:
    - machine_type (L, M, or H)
    - air_temperature (in Kelvin)
    - process_temperature (in Kelvin)
    - rotational_speed (in RPM)
    - torque (in Nm)
    - tool_wear (in minutes)
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # Option to use sample data
    use_sample = st.checkbox("Use sample data instead")
    
    if use_sample:
        # Generate sample data
        np.random.seed(42)
        num_samples = st.slider("Number of sample machines", min_value=5, max_value=50, value=10)
        
        machine_types = np.random.choice(["L", "M", "H"], size=num_samples)
        air_temps = np.random.normal(300, 2, num_samples)
        process_temps = np.random.normal(310, 1, num_samples)
        rotational_speeds = np.random.normal(1500, 100, num_samples)
        torques = np.maximum(30, np.random.normal(40, 5, num_samples))
        tool_wears = np.random.uniform(0, 250, num_samples)
        
        sample_data = pd.DataFrame({
            "machine_type": machine_types,
            "air_temperature": air_temps,
            "process_temperature": process_temps,
            "rotational_speed": rotational_speeds,
            "torque": torques,
            "tool_wear": tool_wears
        })
        
        st.dataframe(sample_data)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    if (uploaded_file is not None or use_sample) and st.button("Run Batch Prediction"):
        with st.spinner("Processing batch prediction..."):
            # Load data
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
            else:
                data = sample_data
            
            # Validate columns
            required_columns = ["machine_type", "air_temperature", "process_temperature", 
                               "rotational_speed", "torque", "tool_wear"]
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return
            
            # Prepare batch data
            machines_data = []
            for _, row in data.iterrows():
                machine_data = {
                    "machine_type": row["machine_type"],
                    "air_temperature": float(row["air_temperature"]),
                    "process_temperature": float(row["process_temperature"]),
                    "rotational_speed": float(row["rotational_speed"]),
                    "torque": float(row["torque"]),
                    "tool_wear": float(row["tool_wear"])
                }
                machines_data.append(machine_data)
            
            # Make batch prediction
            batch_result = make_batch_prediction(machines_data)
            
            if batch_result:
                # Display batch prediction results
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h2 class="sub-header">Batch Prediction Results</h2>', unsafe_allow_html=True)
                
                # Summary
                st.markdown('<h3>Summary</h3>', unsafe_allow_html=True)
                
                summary = batch_result["summary"]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{summary["total_machines"]}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Total Machines</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{summary["failure_count"]}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Predicted Failures</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col3:
                    failure_rate_color = "failure-high" if summary["failure_rate"] > 0.3 else "failure-low"
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value {failure_rate_color}">{summary["failure_rate"]:.1%}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Failure Rate</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Most common failure type
                if "most_common_failure_type" in summary and summary["most_common_failure_type"]:
                    st.markdown(f"**Most Common Failure Type:** {summary['most_common_failure_type']}")
                
                # Detailed results
                st.markdown('<h3>Detailed Results</h3>', unsafe_allow_html=True)
                
                # Create a DataFrame from predictions
                results_data = []
                for i, pred in enumerate(batch_result["predictions"]):
                    # Get original machine data
                    machine_data = machines_data[i]
                    
                    # Get most likely failure type
                    most_likely_type = None
                    if pred["failure_type"]:
                        most_likely_type = max(pred["failure_type"].items(), key=lambda x: x[1])[0]
                    
                    results_data.append({
                        "Machine": i + 1,
                        "Type": machine_data["machine_type"],
                        "Air Temp": f"{machine_data['air_temperature']:.1f} K",
                        "Process Temp": f"{machine_data['process_temperature']:.1f} K",
                        "Speed": f"{machine_data['rotational_speed']:.0f} rpm",
                        "Torque": f"{machine_data['torque']:.1f} Nm",
                        "Tool Wear": f"{machine_data['tool_wear']:.0f} min",
                        "Failure": "Yes" if pred["machine_failure"] else "No",
                        "Probability": f"{pred['failure_probability']:.1%}",
                        "Most Likely Failure": most_likely_type if most_likely_type else "N/A"
                    })
                
                results_df = pd.DataFrame(results_data)
                
                # Style the DataFrame
                def highlight_failures(val):
                    if val == "Yes":
                        return 'background-color: #ffcccc'
                    return ''
                
                styled_df = results_df.style.applymap(highlight_failures, subset=['Failure'])
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Download results as CSV
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="batch_prediction_results.csv",
                    mime="text/csv"
                )
                
                st.markdown('</div>', unsafe_allow_html=True)

def main():
    """
    Main function to run the Streamlit app.
    """
    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/maintenance.png", width=80)
    st.sidebar.title("Navigation")
    
    # Navigation
    page = st.sidebar.radio("Select Page", ["Dashboard", "Single Prediction", "Batch Prediction"])
    
    # Display the selected page
    if page == "Dashboard":
        display_dashboard()
    elif page == "Single Prediction":
        display_single_prediction()
    else:
        display_batch_prediction()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard provides a web interface for the Predictive Maintenance API. "
        "It allows you to make predictions for machine failures and view monitoring statistics."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
